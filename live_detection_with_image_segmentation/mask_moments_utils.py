import cv2
import numpy as np
from matplotlib import pyplot as plt

def rect_to_rotated_box_points(rect_xywh):
    x, y, w, h = float(rect_xywh[0]), float(rect_xywh[1]), float(rect_xywh[2]), float(rect_xywh[3])
    
    center = (x + w / 2, y + h / 2)
    size = (w, h)
    angle = 0.0

    rotated_rect_info = (center, size, angle) 
    
    return cv2.boxPoints(rotated_rect_info).astype(np.int32)

def get_moment_and_rotated_rect(binary_part_mask):
    """
    Обчислює моменти, центроїд та орієнтований мінімальний прямокутник для бінарної маски.
    Повертає кортеж: (moments, centroid_xy, rotated_rect_info, box_points)
    """
    contours, _ = cv2.findContours(binary_part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)

    temp_mask_for_moments = np.zeros_like(binary_part_mask)
    cv2.drawContours(temp_mask_for_moments, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    M = cv2.moments(temp_mask_for_moments)

    if M["m00"] == 0:  
        return None

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    rotated_rect = cv2.minAreaRect(largest_contour)
    
    box = cv2.boxPoints(rotated_rect).astype(np.int32)

    return M, (cX, cY), rotated_rect, box 

def get_fallback_eye_rects(face_rect, left_eye_raw_rect_xywh, right_eye_raw_rect_xywh, img_shape):
    """
    Обчислює або уточнює вісь-вирівняні прямокутники очей, використовуючи face_rect як базову лінію.
    Ця функція призначена для сценаріїв, коли детектор не знайшов одного або обох очей.
    """
    fx, fy, fw, fh = int(face_rect[0]), int(face_rect[1]), int(face_rect[2]), int(face_rect[3])
    img_h, img_w = img_shape[:2]

    target_eye_width = int(fw / 5.0)
    target_eye_height = int(target_eye_width * 0.6)

    center_face_x = fx + fw // 2
    center_face_y = fy + fh // 2

    def create_rect_from_center_and_size(cx, cy, w, h):
        x = max(0, cx - w // 2)
        y = max(0, cy - h // 2)
        w_final = min(w, img_w - x)
        h_final = min(h, img_h - y)
        return [int(x), int(y), int(w_final), int(h_final)]

    adjusted_left_eye_xywh = left_eye_raw_rect_xywh
    adjusted_right_eye_xywh = right_eye_raw_rect_xywh

    if adjusted_left_eye_xywh is not None and adjusted_right_eye_xywh is None:
        lx, ly, lw, lh = int(adjusted_left_eye_xywh[0]), int(adjusted_left_eye_xywh[1]), \
                         int(adjusted_left_eye_xywh[2]), int(adjusted_left_eye_xywh[3])
        center_lx = lx + lw // 2
        center_ly = ly + lh // 2
        dist_x = center_face_x - center_lx 
        mirror_cx = center_face_x + dist_x
        adjusted_right_eye_xywh = create_rect_from_center_and_size(mirror_cx, center_ly, target_eye_width, target_eye_height)

    elif adjusted_right_eye_xywh is not None and adjusted_left_eye_xywh is None:
        rx, ry, rw, rh = int(adjusted_right_eye_xywh[0]), int(adjusted_right_eye_xywh[1]), \
                         int(adjusted_right_eye_xywh[2]), int(adjusted_right_eye_xywh[3])
        center_rx = rx + rw // 2
        center_ry = ry + rh // 2
        dist_x = center_rx - center_face_x
        mirror_cx = center_face_x - dist_x
        adjusted_left_eye_xywh = create_rect_from_center_and_size(mirror_cx, center_ry, target_eye_width, target_eye_height)

    if adjusted_left_eye_xywh is None and adjusted_right_eye_xywh is None:
        eye_y_center = fy + int(fh * 0.3) 
        left_eye_x_center = fx + int(fw * 0.25) 
        right_eye_x_center = fx + int(fw * 0.75) 

        adjusted_left_eye_xywh = create_rect_from_center_and_size(left_eye_x_center, eye_y_center, target_eye_width, target_eye_height)
        adjusted_right_eye_xywh = create_rect_from_center_and_size(right_eye_x_center, eye_y_center, target_eye_width, target_eye_height)
        
    return adjusted_left_eye_xywh, adjusted_right_eye_xywh

def get_fallback_mouth_rect(face_rect, img_shape):
    
    fx, fy, fw, fh = int(face_rect[0]), int(face_rect[1]), int(face_rect[2]), int(face_rect[3])
    img_h, img_w = img_shape[:2]

    mouth_w_ratio = 0.4
    mouth_h_ratio = 0.15
    
    mouth_x_center = fx + fw // 2
    mouth_y_center = fy + int(fh * 0.75) 
    
    mouth_w = int(fw * mouth_w_ratio)
    mouth_h = int(fh * mouth_h_ratio)

    mouth_x = max(0, mouth_x_center - mouth_w // 2)
    mouth_y = max(0, mouth_y_center - mouth_h // 2)
    mouth_w_final = min(mouth_w, img_w - mouth_x)
    mouth_h_final = min(mouth_h, img_h - mouth_y)

    return [int(mouth_x), int(mouth_y), int(mouth_w_final), int(mouth_h_final)]