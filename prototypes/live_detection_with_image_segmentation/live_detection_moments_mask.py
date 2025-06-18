from dataclasses import dataclass
import time
from typing import List, Tuple, Optional, Dict
import cv2
from threading import Thread, Lock
from fastai.vision.all import load_learner
import numpy as np
import pathlib
import torch
from PIL import Image, ImageDraw, ImageFont
import sys
import os
print("Поточний робочий каталог:", os.getcwd())


current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
face_parsing_dir = os.path.join(project_root, 'face_parsing')
print(face_parsing_dir)

if face_parsing_dir not in sys.path:
    sys.path.insert(0, project_root)


from face_parsing.inference import load_model, prepare_image 
from mask_moments_utils import (
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

class DriverFatigueProcessingStream:
    def __init__(self, segmentation_model: FaceSegmentationModel, analyzer: FatigueAnalyzer):
        self.started = False
        self.process_lock = Lock()
        self.frame = None
        self.processed_facial_obj: List[DriverFace] = []
        self.processed_mask: np.ndarray = None 
        self.is_processed = False
        self.segmentation_model = segmentation_model
        self.analyzer = analyzer
        self.FACE_LABELS = [1]; self.BROW_LABELS = [2, 3]; self.EYE_LABELS = [4, 5]
        self.NOSE_LABELS = [10]; self.MOUTH_LABELS = [11, 12, 13]
        self.learn = load_learner('fatigue_model_training\\yawn_eye_model.pkl')
        print("Завантажено класифікатор втоми.")
    
    def start(self):
        if self.started: return None
        self.started = True
        self.thread = Thread(target=self.process, args=())
        self.thread.daemon = True
        self.thread.start()
        return self
    
    def process(self):
        while self.started:
            with self.process_lock:
                if self.is_processed or self.frame is None:
                    continue
                frame_to_process = self.frame.copy()

            detected_faces, frame_mask = self.detect_facial_obj(frame_to_process)
            
            with self.process_lock:
                self.processed_facial_obj = detected_faces
                self.processed_mask = frame_mask
                self.is_processed = True

    def update(self, frame: np.ndarray):
        with self.process_lock:
            self.frame = frame
            self.is_processed = False
    
    def read(self) -> List[DriverFace]:
        with self.process_lock:
            return self.processed_facial_obj.copy()
        
    def read_mask(self) -> Optional[np.ndarray]:
        with self.process_lock:
            return self.processed_mask.copy() if self.processed_mask is not None else None
    
    def stop(self):
        print("Зупинка обробки потоку..."); self.started = False
        if hasattr(self, 'thread') and self.thread.is_alive(): self.thread.join()
        if self.process_lock.locked(): self.process_lock.release()
        print("Потік зупинено.")

    def detect_facial_obj(self, frame: np.ndarray) -> Tuple[List[DriverFace], Optional[np.ndarray]]:
        seg_mask = self.segmentation_model.predict_mask(frame)

        if seg_mask.shape[:2] != frame.shape[:2]: return [], None
        part_masks = {name: np.isin(seg_mask, labels).astype(np.uint8) * 255 for name, labels in {
            'face': self.FACE_LABELS, 'eyes': self.EYE_LABELS, 'brows': self.BROW_LABELS,
            'nose': self.NOSE_LABELS, 'mouth': self.MOUTH_LABELS
        }.items()}
        face_rect = find_part_rect(part_masks['face'])

        if face_rect is None:
            hist_status = self.analyzer.add_and_analyze(0.0, 0.0, 0.0)
            return [], seg_mask 

        left_eye_rect, right_eye_rect = left_right_eye(part_masks['eyes'])
        left_brow_rect, right_brow_rect = left_right_eye(part_masks['brows'])
        nose_rect = find_part_rect(part_masks['nose'])
        mouth_rect = find_part_rect(part_masks['mouth'])
        eye_c = calculate_centroids(part_masks['eyes']); initial_left_c, initial_right_c = (eye_c[0] if len(eye_c)>0 else None), (eye_c[1] if len(eye_c)>1 else None)
        brow_c = calculate_centroids(part_masks['brows']); initial_left_brow_c, initial_right_brow_c = (brow_c[0] if len(brow_c)>0 else None), (brow_c[1] if len(brow_c)>1 else None)
        mouth_c = calculate_centroids(part_masks['mouth']); initial_mouth_c = mouth_c[0] if mouth_c else None
        nose_c = calculate_centroids(part_masks['nose']); initial_nose_c = nose_c[0] if nose_c else None
        (adj_left_eye, adj_right_eye, adj_mouth), (final_left_c, final_right_c, final_mouth_c) = adjust_features_by_proportion(
            face_rect,
            (left_eye_rect, right_eye_rect, initial_left_c, initial_right_c),
            (mouth_rect, initial_mouth_c),
            (left_brow_rect, right_brow_rect, initial_left_brow_c, initial_right_brow_c),
            (nose_rect, initial_nose_c)
        )

        eyes_for_face, mouth_for_face = [], None
        def classify_and_create_driver_part(rect, centroid, original_frame, learner):
            if rect is None or centroid is None: return None, None, None
            x, y, w, h = rect
            if w <= 0 or h <= 0: return None, None, None
            roi = original_frame[y:y+h, x:x+w]
            if roi.size == 0: return None, None, None
            pred_label_orig, _, _ = learner.predict(roi)
            pred_label_orig = str(pred_label_orig)
            ukrainian_label = TRANSLATION_DICT.get(pred_label_orig, pred_label_orig)
            return ukrainian_label, rect_to_box_points(rect), pred_label_orig

        left_eye_label, left_eye_box, left_eye_orig = classify_and_create_driver_part(adj_left_eye, final_left_c, frame, self.learn)
        right_eye_label, right_eye_box, right_eye_orig = classify_and_create_driver_part(adj_right_eye, final_right_c, frame, self.learn)
        mouth_label, mouth_box, mouth_orig = classify_and_create_driver_part(adj_mouth, final_mouth_c, frame, self.learn)

        if left_eye_box is not None: eyes_for_face.append(DriverEye('left_eye', left_eye_box, final_left_c, left_eye_label, left_eye_orig))
        if right_eye_box is not None: eyes_for_face.append(DriverEye('right_eye', right_eye_box, final_right_c, right_eye_label, right_eye_orig))
        if mouth_box is not None: mouth_for_face = DriverMouth(mouth_box, final_mouth_c, mouth_label, mouth_orig)

        left_eye_val = 1.0 if left_eye_orig == "Closed" else 0.0
        right_eye_val = 1.0 if right_eye_orig == "Closed" else 0.0
        mouth_val = 1.0 if mouth_orig in ["yawn", "Open"] else 0.0
        hist_status = self.analyzer.add_and_analyze(left_eye_val, right_eye_val, mouth_val)

        face_centroid = (face_rect[0] + face_rect[2] // 2, face_rect[1] + face_rect[3] // 2)
        faces = [DriverFace(rect_to_box_points(face_rect), face_centroid, TRANSLATION_DICT['Face'],
                                eyes_for_face, mouth_for_face, hist_status)]
        return faces, seg_mask

def draw_info_panel(frame, faces_data, font, panel_width=300):
    h, w, _ = frame.shape
    display_frame = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    display_frame[0:h, 0:w] = frame
    cv2.rectangle(display_frame, (w, 0), (w + panel_width, h), (20, 20, 20), -1)

    if faces_data:
        face = faces_data[0]
        cv2.polylines(display_frame, [face.box_points], True, (0, 255, 0), 2)
        for eye in face.eyes:
            is_closed = eye.original_label == "Closed"
            color = (0, 0, 255) if is_closed else (0, 255, 255)
            cv2.polylines(display_frame, [eye.box_points], True, color, 2)
        if face.mouth:
            is_fatigue_indicator = face.mouth.original_label in ["yawn", "Open"]
            color = (0, 0, 255) if is_fatigue_indicator else (255, 255, 0)
            cv2.polylines(display_frame, [face.mouth.box_points], True, color, 2)

    pil_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    text_x, text_y, line_height = w + 20, 30, 35
    draw.text((text_x, text_y), "СТАТУС ВОДІЯ", font=font, fill=(255, 255, 255))
    text_y += line_height

    if not faces_data:
        draw.text((text_x, h // 2), "Обличчя не знайдено", font=font, fill=(0, 0, 255))
    else:
        face = faces_data[0]

        text_y += line_height
        draw.text((text_x, text_y), "Миттєвий стан:", font=font, fill=(200, 200, 200))
        text_y += line_height

        if face.eyes:
            for eye in face.eyes:
                eye_name = TRANSLATION_DICT.get(eye.id, "Око")
                draw.text((text_x, text_y), f"- {eye_name}: {eye.label}", font=font, fill=(255, 255, 255))
                text_y += line_height
        else:
            draw.text((text_x, text_y), "- Очі не знайдено", font=font, fill=(255, 165, 0))
            text_y += line_height

        mouth_status = face.mouth.label if face.mouth else "Не визначено"
        draw.text((text_x, text_y), f"- Рот: {mouth_status}", font=font, fill=(255, 255, 255))

        text_y += line_height * 1.5
        draw.text((text_x, text_y), "Довгостроковий стан:", font=font, fill=(200, 200, 200))
        text_y += line_height

        hist_status_text = TRANSLATION_DICT.get(face.historical_status, face.historical_status)
        hist_status_color = (255, 0, 0) if face.historical_status == "PROLONGED_FATIGUE" else (0, 255, 0)
        draw.text((text_x, text_y), hist_status_text, font=font, fill=hist_status_color)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

COLOR_LIST = [
    [0, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
    [255, 0, 255],
]

NUM_CLASSES = 19

def create_color_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    msk_color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_index in range(1, NUM_CLASSES + 1):
        if class_index < len(COLOR_LIST):
            msk_color[mask == class_index] = COLOR_LIST[class_index]
    return msk_color


if __name__ == "__main__":
    try:
        font_path = "C:/Windows/Fonts/times.ttf"
        main_font = ImageFont.truetype(font_path, 20)
    except IOError:
        print("Шрифт Arial не знайдено. Буде використано стандартний шрифт.")
        main_font = ImageFont.load_default()

    FPS = 24
    HISTORY_SECONDS = 10
    EYE_PLATEAU_SECONDS = 2.0  
    YAWN_PLATEAU_SECONDS = 2.5 
    FATIGUE_CONFIDENCE = 0.75

    fatigue_analyzer = FatigueAnalyzer(fps=FPS, history_seconds=HISTORY_SECONDS, eye_plateau_sec=EYE_PLATEAU_SECONDS, yawn_plateau_sec=YAWN_PLATEAU_SECONDS, fatigue_confidence=FATIGUE_CONFIDENCE)
    seg_model_instance = FaceSegmentationModel(model_name="resnet34", weight_path="prototypes\\live_detection_with_image_segmentation\\weights\\resnet34.pt", input_size=(512, 512))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Помилка: не вдалося відкрити вебкамеру.")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    dfps = DriverFatigueProcessingStream(segmentation_model=seg_model_instance, analyzer=fatigue_analyzer).start()

    display_mode = 0  
    mode_names = ["Основний вигляд (інфо-панель)", "Вигляд маски сегментації"]
    print("\nСистема запущена. Натисніть 'q' для виходу, 'w' для перемикання вигляду.")
    print(f"Поточний режим: {mode_names[display_mode]}")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Не вдалося отримати кадр. Завершення роботи.")
            break
        
        frame = cv2.flip(frame, 1) 

        dfps.update(frame.copy())
        
        current_faces = dfps.read()
        frame_mask = dfps.read_mask()

        if display_mode == 0:
            display_frame = draw_info_panel(frame, current_faces, main_font)
        
        elif display_mode == 1:
            if frame_mask is not None:
                color_mask_img = create_color_mask(frame_mask)
                display_frame = cv2.addWeighted(frame, 0.6, color_mask_img, 0.4, 0)
            else:
                display_frame = frame
        
        cv2.imshow("Система моніторингу водія", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'): 
            display_mode = (display_mode + 1) % len(mode_names)
            print(f"Режим змінено на: {mode_names[display_mode]}")

    dfps.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Роботу завершено.")