import cv2
import numpy as np
from matplotlib import pyplot as plt

def find_part_rect(part_seg_mask):
    cnt, _ = cv2.findContours(part_seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnt) < 1:
        return np.asarray([0,0,0,0])
    print(cnt[0].min(axis=0), cnt[0].max(axis=0))
    return np.asarray(cv2.boundingRect(cnt[0][:,0,:]))

def left_right_eye(seg_mask_eyes):
    cnt, _ = cv2.findContours(seg_mask_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnt) == 2:
        eyes_rects = [np.asarray(cv2.boundingRect(cnt[i][:,0,:])) for i in range(2)]
        eyes_rects = np.sort(eyes_rects, axis=0)
        print(eyes_rects)
        return eyes_rects
    
    elif len(cnt) == 1:
        eye_rect = np.asarray(cv2.boundingRect(cnt[0][:, 0, :]))
        print("Found 1 eye:", eye_rect)
        return eye_rect, None

    else:
        print(f"Found {len(cnt)} eyes — unsupported number.")
        return None, None
    
def adjust_eye_rects_by_face(face_rect, left_eye_raw_rect, right_eye_raw_rect, img_shape):
    fx, fy, fw, fh = face_rect
    img_h, img_w = img_shape[:2]

    target_eye_width = int(fw / 5.0)
    target_eye_height = int(target_eye_width * 0.6)

    center_face_x = fx + fw // 2
    center_face_y = fy + fh // 2

    def adjust_eye(raw_rect):
        if raw_rect is None or raw_rect[2] <= 0 or raw_rect[3] <= 0:
            return None
        x, y, w, h = raw_rect
        cx = x + w // 2
        cy = y + h // 2
        ax = max(0, cx - target_eye_width // 2)
        ay = max(0, cy - target_eye_height // 2)
        return np.array([
            ax,
            ay,
            min(target_eye_width, img_w - ax),
            min(target_eye_height, img_h - ay)
        ])

    adjusted_left_eye = adjust_eye(left_eye_raw_rect)
    adjusted_right_eye = adjust_eye(right_eye_raw_rect)

    if adjusted_left_eye is not None and adjusted_right_eye is None:
        lx, ly, lw, lh = adjusted_left_eye
        center_lx = lx + lw // 2
        dist = center_face_x - center_lx
        mirror_cx = center_face_x + dist
        ax = max(0, mirror_cx - target_eye_width // 2)
        ay = ly
        adjusted_right_eye = np.array([
            ax,
            ay,
            min(target_eye_width, img_w - ax),
            min(target_eye_height, img_h - ay)
        ])

    elif adjusted_right_eye is not None and adjusted_left_eye is None:
        rx, ry, rw, rh = adjusted_right_eye
        center_rx = rx + rw // 2
        dist = center_rx - center_face_x
        mirror_cx = center_face_x - dist
        ax = max(0, mirror_cx - target_eye_width // 2)
        ay = ry
        adjusted_left_eye = np.array([
            ax,
            ay,
            min(target_eye_width, img_w - ax),
            min(target_eye_height, img_h - ay)
        ])

    if adjusted_left_eye is None and adjusted_right_eye is None:
        eye_y = fy + int(fh * 0.3)
        left_eye_x = fx + int(fw * 0.25) - target_eye_width // 2
        right_eye_x = fx + int(fw * 0.75) - target_eye_width // 2

        adjusted_left_eye = np.array([
            max(0, left_eye_x),
            max(0, eye_y),
            min(target_eye_width, img_w - left_eye_x),
            min(target_eye_height, img_h - eye_y)
        ])

        adjusted_right_eye = np.array([
            max(0, right_eye_x),
            max(0, eye_y),
            min(target_eye_width, img_w - right_eye_x),
            min(target_eye_height, img_h - eye_y)
        ])

    return adjusted_left_eye, adjusted_right_eye
