#!/usr/bin/env python

import time
from dataclasses import dataclass
from threading import Thread, Lock
from typing import List
import cv2
import numpy as np
import pathlib
import torch
from PIL import Image
from fastai.vision.all import load_learner
import sys
import os

pathlib.PosixPath = pathlib.WindowsPath

learn = load_learner('fatigue_model_training\\yawn_eye_model.pkl')

current_dir = os.path.dirname(os.path.abspath(__file__))
submodule_dir = os.path.join(current_dir, 'face_parsing')

if submodule_dir not in sys.path:
    sys.path.insert(0, submodule_dir)

from face_parsing.inference import load_model, prepare_image 

# from inference import load_model, prepare_image
from mask_interpeter import find_part_rect, left_right_eye, adjust_eye_rects_by_face


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model = load_model("resnet34", 19, "live_detection_with_image_segmentation\\weights\\resnet34.pt", device)


def predict_mask(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_size = image.size
    image_batch = prepare_image(image).to(device)
    with torch.no_grad():
        output = seg_model(image_batch)[0]
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)
    mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))
    restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)
    return np.array(restored_mask)


@dataclass
class DriverEye:
    roi: np.ndarray
    label: str


@dataclass
class DriverFace:
    roi: np.ndarray
    label: str
    eyes: List[DriverEye]


class DriverFatigueProcessingStream:
    def __init__(self):
        self.started = False
        self.process_lock = Lock()
        self.frame = None
        self.facial_msk = None
        self.facial_obj = []
        self.is_processed = False

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.process, args=())
        self.thread.start()
        return self

    def process(self):
        while self.started:
            with self.process_lock:
                if self.is_processed or self.frame is None:
                    continue
                frame_buff = self.frame.copy()

            self.facial_msk, self.facial_obj = self.detect_facial_obj(frame_buff)
            self.is_processed = True    

    def update(self, frame):
        with self.process_lock:
            self.frame = frame
            self.is_processed = False
        return self.facial_msk, self.facial_obj

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        if self.process_lock.locked():
            self.process_lock.release()

    @staticmethod
    def detect_facial_obj(frame):
        seg_mask = predict_mask(frame)
        seg_mask_face = (seg_mask == 1).astype(np.uint8) * 255

        face_rect = find_part_rect(seg_mask_face)
        if face_rect is None:
            return []

        x, y, w, h = face_rect
        if w == 0 or h == 0:
            return []

        face_roi = frame[y:y+h, x:x+w]
        pred_face, _, _ = learn.predict(face_roi)

        seg_mask_eyes = ((seg_mask == 4) | (seg_mask == 5)).astype(np.uint8) * 255
        left_eye_rect, right_eye_rect = left_right_eye(seg_mask_eyes)

        if left_eye_rect is None:
            left_eye_rect = (0, 0, 0, 0)
        if right_eye_rect is None:
            right_eye_rect = (0, 0, 0, 0)

        left_eye_rect, right_eye_rect = adjust_eye_rects_by_face(left_eye_rect, right_eye_rect, (x, y, w, h), frame.shape)

        eyes = []
        for eye_rect in [left_eye_rect, right_eye_rect]:
            ex, ey, ew, eh = eye_rect
            if ew == 0 or eh == 0:
                continue
            eye_roi = frame[ey:ey+eh, ex:ex+ew]
            pred_eye, _, _ = learn.predict(eye_roi)
            eyes.append(DriverEye((ex - x, ey - y, ew, eh), str(pred_eye)))

        return seg_mask, [DriverFace((x, y, w, h), str(pred_face), eyes)]


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

num_classes = 19 

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    dfps = DriverFatigueProcessingStream().start()

    if not cap.isOpened():
        print("Помилка: Не вдалося відкрити камеру.")
        sys.exit(1)

    display_mode = 2 
    mode_names = ["Лише прямокутники", "Лише маска", "Прямокутники та маска"]
    print(f"Початковий режим відображення: {mode_names[display_mode]} (натисніть 'w' для перемикання).")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Помилка: Не вдалося отримати кадр")
            break

        msk, faces = dfps.update(frame.copy()) 
        
        display_frame = frame.copy()

        if display_mode == 0 or display_mode == 2:
            for face in faces:
                x, y, w, h = face.roi
                if w > 0 and h > 0 and x >= 0 and y >= 0 and (x+w) <= frame.shape[1] and (y+h) <= frame.shape[0]:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                    cv2.putText(
                        display_frame,
                        face.label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255), 
                        2,
                    )

                for eye in face.eyes:
                    ex, ey, ew, eh = eye.roi
                    abs_ex = x + ex
                    abs_ey = y + ey
                    if ew > 0 and eh > 0 and abs_ex >= 0 and abs_ey >= 0 and \
                       (abs_ex + ew) <= frame.shape[1] and (abs_ey + eh) <= frame.shape[0]:
                        cv2.rectangle(
                            display_frame,
                            (abs_ex, abs_ey),
                            (abs_ex + ew, abs_ey + eh),
                            (0, 255, 255), 
                            2,
                        )
                        cv2.putText(
                            display_frame,
                            eye.label,
                            (abs_ex, abs_ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0), 
                            2,
                        )

        if (display_mode == 1 or display_mode == 2) and msk is not None:
            msk_color = np.zeros((msk.shape[0], msk.shape[1], 3), dtype=np.uint8)

            for class_index in range(1, num_classes + 1): 
                if class_index < len(COLOR_LIST): 
                    msk_color[msk == class_index] = COLOR_LIST[class_index]

            display_frame = cv2.addWeighted(display_frame, 0.6, msk_color, 0.4, 0)

        cv2.imshow("Driver Monitoring System", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print("Відеопотік зупинено користувачем.")
            break
        elif key == ord("w"): 
            display_mode = (display_mode + 1) % 3 
            print(f"Режим відображення: {mode_names[display_mode]} (натисніть 'w' для перемикання).")

    dfps.stop()
    cap.release()
    cv2.destroyAllWindows()