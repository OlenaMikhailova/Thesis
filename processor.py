import logging
from dataclasses import dataclass
import pathlib
from threading import Lock, Thread
from typing import Any, List, Optional, Tuple
from fastai.vision.all import load_learner

import cv2
import numpy as np
import torch
from PIL import Image

from face_parsing.inference import load_model, prepare_image
from face_parsing.models.bisenet import BiSeNet

logger = logging.getLogger("Driver Fatigue Processor")
logging.basicConfig(level=logging.INFO)

from prototypes.live_detection_with_image_segmentation.mask_moments_utils import (
    calculate_centroids,
    find_part_rect,
    left_right_eye,
    adjust_features_by_proportion
)

TRANSLATION_DICT = {
    "Open": "Відкритий",
    "Closed": "Закрите",
    "yawn": "Позіхання",
    "no_yawn": "Спокійний",
    "Face": "Обличчя",
    "left_eye": "Ліве око",
    "right_eye": "Праве око",
    "ANALYZING": "Аналіз...",
    "NORMAL": "Норма",
    "PROLONGED_FATIGUE": "ТРИВАЛА ВТОМА!"
}

def rect_to_box_points(rect):
    if rect is None: return None
    x, y, w, h = rect
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)

if hasattr(pathlib, 'PosixPath'):
    pathlib.PosixPath = pathlib.WindowsPath

class FatigueAnalyzer:
    def __init__(self, fps, history_seconds, eye_plateau_sec, yawn_plateau_sec, fatigue_confidence=0.75):
        self.FPS = fps
        self.FRAMES_TO_REMEMBER = int(fps * history_seconds)
        self.EYE_PLATEAU_FRAMES = int(fps * eye_plateau_sec)
        self.YAWN_PLATEAU_FRAMES = int(fps * yawn_plateau_sec)
        self.FATIGUE_CONFIDENCE_THRESHOLD = fatigue_confidence
        
        self.queue = np.zeros((self.FRAMES_TO_REMEMBER, 3), dtype=np.float16)
        self.WARMUP_FRAMES = self.FRAMES_TO_REMEMBER // 2
        self.frame_idx = 0
        print(f"Ініціалізовано аналізатор втоми: Плато для очей={eye_plateau_sec}с, Плато для рота={yawn_plateau_sec}с")
        print(f"Поріг спрацювання втоми: {int(self.FATIGUE_CONFIDENCE_THRESHOLD * 100)}% кадрів у періоді")

    def add_and_analyze(self, left_eye_closed, right_eye_closed, is_mouth_fatigue_indicator) -> str:
        self.queue = np.roll(self.queue, shift=-1, axis=0)
        self.queue[-1, :] = [left_eye_closed, right_eye_closed, is_mouth_fatigue_indicator]

        if self.frame_idx < self.WARMUP_FRAMES:
            self.frame_idx += 1
            return "ANALYZING"

        is_prolonged_fatigue = False
        
        eye_fatigue_frames_needed = self.EYE_PLATEAU_FRAMES * self.FATIGUE_CONFIDENCE_THRESHOLD
        
        if (np.sum(self.queue[-self.EYE_PLATEAU_FRAMES:, 0]) >= eye_fatigue_frames_needed or
            np.sum(self.queue[-self.EYE_PLATEAU_FRAMES:, 1]) >= eye_fatigue_frames_needed):
            is_prolonged_fatigue = True
            
        mouth_fatigue_frames_needed = self.YAWN_PLATEAU_FRAMES * self.FATIGUE_CONFIDENCE_THRESHOLD
        
        if np.sum(self.queue[-self.YAWN_PLATEAU_FRAMES:, 2]) >= mouth_fatigue_frames_needed:
            is_prolonged_fatigue = True

        return "PROLONGED_FATIGUE" if is_prolonged_fatigue else "NORMAL"

@dataclass
class DriverEye:
    id: str
    box_points: np.ndarray
    centroid: Tuple[int, int]
    label: str
    original_label: str

@dataclass
class DriverMouth:
    box_points: np.ndarray
    centroid: Tuple[int, int]
    label: str
    original_label: str

@dataclass
class DriverFace:
    box_points: np.ndarray
    centroid: Tuple[int, int]
    label: str
    eyes: List[DriverEye]
    mouth: Optional[DriverMouth]
    historical_status: str

class FaceSegmentationModel:
    def __init__(self, model_name: str, weight_path: str, input_size: Tuple[int, int] = (512, 512)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.num_classes = 19
        self.model = load_model(model_name, self.num_classes, weight_path, self.device)
        print(f"Модель для сегментації обличчя завантажена на {self.device}")

    @torch.no_grad()
    def predict_mask(self, frame: np.ndarray) -> np.ndarray:
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        original_h, original_w = frame.shape[:2]
        image_batch_tensor = prepare_image(image_pil, self.input_size).to(self.device)
        output = self.model(image_batch_tensor)[0]
        predicted_mask_resized = output.squeeze(0).cpu().numpy().argmax(0)
        predicted_mask = cv2.resize(predicted_mask_resized.astype(np.uint8),
                                     (original_w, original_h),
                                     interpolation=cv2.INTER_NEAREST)
        return predicted_mask


@dataclass
class DriverFatigueResult:
    mask: Optional[np.typing.NDArray] = None
    faces: List[DriverFace] = None


class DriverFatigueProcessor:
    def __init__(self, segmentation_model: FaceSegmentationModel, device: torch.device, analyzer: FatigueAnalyzer):
        self._seg_model = segmentation_model
        self._device = device
        self.analyzer = analyzer
        self.learn = load_learner('fatigue_model_training/yawn_eye_model.pkl')
        self.FACE_LABELS  = [1]
        self.EYE_LABELS   = [4, 5]
        self.BROW_LABELS  = [2, 3]
        self.NOSE_LABELS  = [10]
        self.MOUTH_LABELS = [11, 12, 13]

    def process(self, frame):
        mask = self.get_frame_mask(frame)
        faces = self.get_frame_rois(frame, mask)
        return DriverFatigueResult(mask, faces=faces)

    def get_frame_mask(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        original_size = image.size
        image_batch = prepare_image(image).to(self._device)
        with torch.no_grad():
            output = self._seg_model(image_batch)[0]
            predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)
        mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))
        restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)
        return np.array(restored_mask)

    def get_frame_rois(self, frame: np.ndarray, mask: np.ndarray) -> List[DriverFace]:
        part_masks = { 
            'face':  np.isin(mask, self.FACE_LABELS).astype(np.uint8)*255,
            'eyes':  np.isin(mask, self.EYE_LABELS).astype(np.uint8)*255,
            'brows': np.isin(mask, self.BROW_LABELS).astype(np.uint8)*255,
            'nose':  np.isin(mask, self.NOSE_LABELS).astype(np.uint8)*255,
            'mouth': np.isin(mask, self.MOUTH_LABELS).astype(np.uint8)*255,
        }

        face_rect = find_part_rect(part_masks['face'])
        if face_rect is None:
            _ = self.analyzer.add_and_analyze(0, 0, 0)
            return []

        left_eye_rect,  right_eye_rect  = left_right_eye(part_masks['eyes'])
        left_brow_rect, right_brow_rect = left_right_eye(part_masks['brows'])
        nose_rect    = find_part_rect(part_masks['nose'])
        mouth_rect   = find_part_rect(part_masks['mouth'])

        eye_cs   = calculate_centroids(part_masks['eyes'])
        brow_cs  = calculate_centroids(part_masks['brows'])
        nose_cs  = calculate_centroids(part_masks['nose'])
        mouth_cs = calculate_centroids(part_masks['mouth'])

        (adj_le, adj_re, adj_m), (c_le, c_re, c_m) = adjust_features_by_proportion(
            face_rect,
            (left_eye_rect, right_eye_rect,  eye_cs[0]   if len(eye_cs)>0   else None,
                                     eye_cs[1]   if len(eye_cs)>1   else None),
            (mouth_rect,               mouth_cs[0] if mouth_cs else None),
            (left_brow_rect, right_brow_rect, brow_cs[0]  if len(brow_cs)>0 else None,
                                               brow_cs[1]  if len(brow_cs)>1 else None),
            (nose_rect,                nose_cs[0]  if nose_cs  else None)
        )

        # 5) Функція для класифікації ROI через fastai
        def classify(rect, centroid):
            if rect is None or centroid is None:
                return None, None, None
            x, y, w, h = rect
            roi = frame[y:y+h, x:x+w]
            pred_orig, _, _ = self.learn.predict(roi)
            orig = str(pred_orig)
            ua   = TRANSLATION_DICT.get(orig, orig)
            return ua, rect_to_box_points(rect), orig

        eyes = []
        le_lbl, le_box, le_orig = classify(adj_le, c_le)
        re_lbl, re_box, re_orig = classify(adj_re, c_re)
        m_lbl,  m_box,  m_orig  = classify(adj_m,  c_m)

        if le_box is not None:
            eyes.append(DriverEye('left_eye', le_box, c_le, le_lbl, le_orig))
        if re_box is not None:
            eyes.append(DriverEye('right_eye', re_box, c_re, re_lbl, re_orig))

        mouth = None
        if m_box is not None:
            mouth = DriverMouth(m_box, c_m, m_lbl, m_orig)


        le_val = 1.0 if le_orig == "Closed" else 0.0
        re_val = 1.0 if re_orig == "Closed" else 0.0
        m_val  = 1.0 if m_orig in ["yawn", "Open"] else 0.0
        hist_status = self.analyzer.add_and_analyze(le_val, re_val, m_val)

        x, y, w, h = face_rect
        centroid = (x + w//2, y + h//2)
        face = DriverFace(
            box_points        = rect_to_box_points(face_rect),
            centroid          = centroid,
            label             = TRANSLATION_DICT['Face'],
            eyes              = eyes,
            mouth             = mouth,
            historical_status = hist_status
        )

        return [face]


class DriverFatigueRunner:
    def __init__(self, processor: DriverFatigueProcessor):
        self.started = False
        self.process_lock = Lock()

        self.processor = processor

        self.frame = None
        self.frame_result = None
        self.is_processed = False

    def start(self):
        if self.started:
            logger.warning("Runner is already started.")
            return self
        self.started = True
        self.thread = Thread(target=self.run, args=())
        self.thread.start()
        return self

    def run(self):
        while self.started:
            with self.process_lock:
                if self.frame is None or self.is_processed:
                    continue
                frame_buff = self.frame.copy()
            self.frame_result = self.processor.process(frame_buff)
            self.is_processed = True

    def update(self, frame):
        with self.process_lock:
            self.frame = frame
            self.is_processed = False
        return self.frame_result

    def stop(self):
        if self.started:
            self.started = False
            if self.thread.is_alive():
                self.thread.join()
            if self.process_lock.locked():
                self.process_lock.release()
