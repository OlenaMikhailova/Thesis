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

pathlib.PosixPath = pathlib.WindowsPath

learn = load_learner('fatigue_model_training\\yawn_eye_model.pkl')

from inference import load_model, prepare_image
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
        self.facial_obj = []

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
                if self.frame is not None:
                    self.facial_obj = self.detect_facial_obj(self.frame)
            time.sleep(0.01)

    def update(self, frame):
        with self.process_lock:
            self.frame = frame
        return self.facial_obj

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

        return [DriverFace((x, y, w, h), str(pred_face), eyes)]



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    dfps = DriverFatigueProcessingStream().start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Помилка: Не вдалося отримати кадр")
            break

        faces = dfps.update(frame.copy())

        for face in faces:
            x, y, w, h = face.roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, face.label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            for eye in face.eyes:
                ex, ey, ew, eh = eye.roi
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 255), 2)
                cv2.putText(frame, eye.label, (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Driver Monitoring System", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Відеопотік зупинено користувачем.")
            break

    dfps.stop()
    cap.release()
    cv2.destroyAllWindows()
