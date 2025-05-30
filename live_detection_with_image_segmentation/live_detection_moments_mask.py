from dataclasses import dataclass
import time
from typing import List, Tuple, Optional
import cv2
from threading import Thread, Lock
from fastai.vision.all import load_learner
import numpy as np
import pathlib
import torch
from PIL import Image

from inference import load_model, prepare_image 

from mask_moments_utils import (
    rect_to_rotated_box_points,
    get_moment_and_rotated_rect,
    get_fallback_eye_rects,
    get_fallback_mouth_rect
)

pathlib.PosixPath = pathlib.WindowsPath

@dataclass
class DriverEye:
    box_points: np.ndarray
    centroid: Tuple[int, int]
    label: str

@dataclass
class DriverFace:
    box_points: np.ndarray
    centroid: Tuple[int, int]
    label: str
    eyes: List[DriverEye]

class FaceSegmentationModel:
    def __init__(self, model_name: str, weight_path: str, input_size: Tuple[int, int] = (512, 512)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.num_classes = 19 

        self.model = load_model(model_name, self.num_classes, weight_path, self.device)

    @torch.no_grad()
    def predict_mask(self, frame: np.ndarray) -> np.ndarray:
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        original_h, original_w = frame.shape[:2]

        image_batch_tensor = prepare_image(image_pil, self.input_size).to(self.device)

        # Run inference
        output = self.model(image_batch_tensor)[0] 
        predicted_mask_resized = output.squeeze(0).cpu().numpy().argmax(0)

        predicted_mask = cv2.resize(predicted_mask_resized.astype(np.uint8), 
                                     (original_w, original_h), 
                                     interpolation=cv2.INTER_NEAREST)
        
        return predicted_mask

class DriverFatigueProcessingStream:
    def __init__(self, segmentation_model: FaceSegmentationModel):
        self.started = False
        self.process_lock = Lock()
        self.frame = None
        self.facial_obj: List[DriverFace] = []
        self.segmentation_model = segmentation_model

        self.EYE_LABELS = [4, 5] 
        self.MOUTH_LABELS = [11, 12, 13] 

        self.learn = load_learner('fatigue_model_training\\yawn_eye_model.pkl')

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = Thread(target=self.process, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def process(self):
        while self.started:
            self.process_lock.acquire()
            if self.frame is not None:
                self.facial_obj = self.detect_facial_obj(
                    self.frame.copy() 
                )
            self.process_lock.release()
            time.sleep(0.01)

    def update(self, frame) -> List[DriverFace]:
        self.process_lock.acquire()
        self.frame = frame
        current_facial_obj = self.facial_obj.copy() 
        self.process_lock.release()
        return current_facial_obj

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        if self.process_lock.locked():
            self.process_lock.release()

    def detect_facial_obj(self, frame: np.ndarray) -> List[DriverFace]:
        seg_mask = self.segmentation_model.predict_mask(frame)
        
        if seg_mask is None or seg_mask.shape[:2] != frame.shape[:2]:
            return []
        
        faces = []

        seg_mask_face_binary = np.zeros_like(seg_mask, dtype=np.uint8)
        seg_mask_face_binary[seg_mask != 0] = 255 

        seg_mask_eyes_binary = np.zeros_like(seg_mask, dtype=np.uint8)
        for label in self.EYE_LABELS:
            seg_mask_eyes_binary[seg_mask == label] = 255 

        seg_mask_mouth_binary = np.zeros_like(seg_mask, dtype=np.uint8)
        for label in self.MOUTH_LABELS:
            seg_mask_mouth_binary[seg_mask == label] = 255 
        
        
        face_rect_xywh = None
        face_contours, _ = cv2.findContours(seg_mask_face_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if face_contours:
            largest_face_contour = max(face_contours, key=cv2.contourArea)
            face_rect_xywh = cv2.boundingRect(largest_face_contour)
        
        left_eye_final_data = None
        right_eye_final_data = None
        mouth_final_data = None
        face_final_data = None

        if face_rect_xywh is not None:
            face_centroid = (face_rect_xywh[0] + face_rect_xywh[2] // 2, face_rect_xywh[1] + face_rect_xywh[3] // 2)
            face_box_points = rect_to_rotated_box_points(face_rect_xywh)
            face_final_data = (face_box_points, face_centroid, (None, (face_rect_xywh[2], face_rect_xywh[3]), 0.0))

            eye_contours_found, _ = cv2.findContours(seg_mask_eyes_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected_eyes_full_info = []

            for contour in eye_contours_found:
                temp_eye_mask = np.zeros_like(seg_mask_eyes_binary)
                cv2.drawContours(temp_eye_mask, [contour], -1, 255, thickness=cv2.FILLED)
                part_info = get_moment_and_rotated_rect(temp_eye_mask)
                if part_info:
                    detected_eyes_full_info.append(part_info)

            if len(detected_eyes_full_info) == 2:
                detected_eyes_full_info.sort(key=lambda x: x[1][0]) 
                left_eye_final_data = (detected_eyes_full_info[0][3], detected_eyes_full_info[0][1], detected_eyes_full_info[0][2])
                right_eye_final_data = (detected_eyes_full_info[1][3], detected_eyes_full_info[1][1], detected_eyes_full_info[1][2])

            elif len(detected_eyes_full_info) == 1:
                detected_eye_full_info = detected_eyes_full_info[0]
                detected_eye_xywh = cv2.boundingRect(detected_eye_full_info[3])
                
                left_eye_detected_for_fallback_xywh = None
                right_eye_detected_for_fallback_xywh = None
                if detected_eye_xywh[0] < face_rect_xywh[0] + face_rect_xywh[2] / 2:
                    left_eye_detected_for_fallback_xywh = detected_eye_xywh
                    left_eye_final_data = (detected_eye_full_info[3], detected_eye_full_info[1], detected_eye_full_info[2])
                else:
                    right_eye_detected_for_fallback_xywh = detected_eye_xywh
                    right_eye_final_data = (detected_eye_full_info[3], detected_eye_full_info[1], detected_eye_full_info[2])

                inferred_left_eye_xywh, inferred_right_eye_xywh = get_fallback_eye_rects(
                    face_rect_xywh, 
                    left_eye_detected_for_fallback_xywh, 
                    right_eye_detected_for_fallback_xywh, 
                    frame.shape
                )
                
                if left_eye_final_data is None and inferred_left_eye_xywh is not None:
                    centroid_l = (inferred_left_eye_xywh[0] + inferred_left_eye_xywh[2]//2, inferred_left_eye_xywh[1] + inferred_left_eye_xywh[3]//2)
                    size_l = (inferred_left_eye_xywh[2], inferred_left_eye_xywh[3])
                    left_eye_final_data = (rect_to_rotated_box_points(inferred_left_eye_xywh), centroid_l, (None, size_l, 0.0))
                
                if right_eye_final_data is None and inferred_right_eye_xywh is not None:
                    centroid_r = (inferred_right_eye_xywh[0] + inferred_right_eye_xywh[2]//2, inferred_right_eye_xywh[1] + inferred_right_eye_xywh[3]//2)
                    size_r = (inferred_right_eye_xywh[2], inferred_right_eye_xywh[3])
                    right_eye_final_data = (rect_to_rotated_box_points(inferred_right_eye_xywh), centroid_r, (None, size_r, 0.0))

            elif len(detected_eyes_full_info) == 0:
                inferred_left_eye_xywh, inferred_right_eye_xywh = get_fallback_eye_rects(face_rect_xywh, None, None, frame.shape)
                
                centroid_l = (inferred_left_eye_xywh[0] + inferred_left_eye_xywh[2]//2, inferred_left_eye_xywh[1] + inferred_left_eye_xywh[3]//2)
                size_l = (inferred_left_eye_xywh[2], inferred_left_eye_xywh[3])
                left_eye_final_data = (rect_to_rotated_box_points(inferred_left_eye_xywh), centroid_l, (None, size_l, 0.0))

                centroid_r = (inferred_right_eye_xywh[0] + inferred_right_eye_xywh[2]//2, inferred_right_eye_xywh[1] + inferred_right_eye_xywh[3]//2)
                size_r = (inferred_right_eye_xywh[2], inferred_right_eye_xywh[3])
                right_eye_final_data = (rect_to_rotated_box_points(inferred_right_eye_xywh), centroid_r, (None, size_r, 0.0))

            mouth_detected_full_info = get_moment_and_rotated_rect(seg_mask_mouth_binary)
            if mouth_detected_full_info:
                mouth_final_data = (mouth_detected_full_info[3], mouth_detected_full_info[1], mouth_detected_full_info[2])
            elif face_rect_xywh is not None:
                inferred_mouth_xywh = get_fallback_mouth_rect(face_rect_xywh, frame.shape)
                centroid_m = (inferred_mouth_xywh[0] + inferred_mouth_xywh[2]//2, inferred_mouth_xywh[1] + inferred_mouth_xywh[3]//2)
                size_m = (inferred_mouth_xywh[2], inferred_mouth_xywh[3])
                mouth_final_data = (rect_to_rotated_box_points(inferred_mouth_xywh), centroid_m, (None, size_m, 0.0))

            eyes_for_face = []
            
            def classify_and_create_driver_part(part_data, original_frame, fastai_learner_instance):
                if part_data:
                    box_points = part_data[0]
                    centroid = part_data[1]

                    x, y, w, h = cv2.boundingRect(box_points)
                    
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, original_frame.shape[1] - x)
                    h = min(h, original_frame.shape[0] - y)

                    if w > 0 and h > 0:
                        roi = original_frame[y:y+h, x:x+w]
                        pred_label, _, _ = fastai_learner_instance.predict(roi)
                        return box_points, centroid, str(pred_label)
                return None, None, None

            left_eye_box_points, left_eye_centroid, left_eye_label = classify_and_create_driver_part(left_eye_final_data, frame, self.learn)
            if left_eye_box_points is not None:
                eyes_for_face.append(DriverEye(left_eye_box_points, left_eye_centroid, left_eye_label))

            right_eye_box_points, right_eye_centroid, right_eye_label = classify_and_create_driver_part(right_eye_final_data, frame, self.learn)
            if right_eye_box_points is not None:
                eyes_for_face.append(DriverEye(right_eye_box_points, right_eye_centroid, right_eye_label))

            face_box_points_final, face_centroid_final, face_label_fastai = classify_and_create_driver_part(face_final_data, frame, self.learn)
            
            if face_box_points_final is not None:
                faces.append(DriverFace(face_box_points_final, face_centroid_final, face_label_fastai, eyes_for_face))

        return faces


if __name__ == "__main__":
    seg_model_instance = FaceSegmentationModel(
        model_name="resnet34", 
        weight_path="live_detection_with_image_segmentation\\weights\\resnet34.pt", 
        input_size=(512, 512) 
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    dfps = DriverFatigueProcessingStream(segmentation_model=seg_model_instance).start()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Помилка: Не вдалося отримати кадр")
            break

        current_faces = dfps.update(frame.copy()) 

        for face in current_faces:
            cv2.polylines(frame, [face.box_points], True, (0, 255, 0), 2) 
            cv2.putText(frame, face.label, (face.centroid[0], face.centroid[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.drawMarker(frame, face.centroid, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

            for eye in face.eyes:
                cv2.polylines(frame, [eye.box_points], True, (0, 255, 255), 2) 
                cv2.putText(frame, eye.label, (eye.centroid[0], eye.centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.drawMarker(frame, eye.centroid, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
        
        cv2.imshow("Driver Monitoring System", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Відеопотік зупинено користувачем.")
            break

    dfps.stop()
    cap.release()
    cv2.destroyAllWindows()