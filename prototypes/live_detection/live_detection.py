from dataclasses import dataclass
import time
from typing import List
import cv2
from threading import Thread, Lock
from fastai.vision.all import load_learner
import numpy as np
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
learn = load_learner('fatigue_model_training\\yawn_eye_model.pkl')


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
            self.process_lock.acquire()
            if self.frame is not None:
                self.facial_obj = DriverFatigueProcessingStream.detect_facial_obj(self.frame)
            self.process_lock.release()
            time.sleep(0.01)

    def update(self, frame):
        self.process_lock.acquire()
        self.frame = frame
        self.process_lock.release()
        return self.facial_obj

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        if self.process_lock.locked():
            self.process_lock.release()

    @staticmethod
    def detect_facial_obj(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = []
        faces_rois = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces_rois:
            pred_face, _, _ = learn.predict(frame[y:y + h, x:x + w])

            eyes = []
            eyes_rois = eye_cascade.detectMultiScale(gray[y:y + h, x:x + w], minNeighbors=10)
            for (ex, ey, ew, eh) in eyes_rois:
                pred_eye, _, _ = learn.predict(frame[y + ey:y + ey + eh, x + ex:x + ex + ew])
                eyes.append(DriverEye((ex, ey, ew, eh), str(pred_eye)))

            faces.append(DriverFace((x, y, w, h), str(pred_face), eyes))

        return faces


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
        print(faces)

        for face in faces:
            x, y, w, h = face.roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, face.label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            for eye in face.eyes:
                (ex, ey, ew, eh) = eye.roi
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 255), 2)
                cv2.putText(frame, eye.label, (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Driver Monitoring System", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Відеопотік зупинено користувачем.")
            break

    dfps.stop()
    cap.release()
    cv2.destroyAllWindows()