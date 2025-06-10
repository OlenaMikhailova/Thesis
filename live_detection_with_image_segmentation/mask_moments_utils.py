import cv2
import numpy as np
from matplotlib import pyplot as plt

def find_part_rect(part_seg_mask):
    contours, _ = cv2.findContours(part_seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 1:
        return np.asarray([0, 0, 0, 0])
    x, y, w, h = cv2.boundingRect(contours[0][:,0,:]) 
    
    return np.asarray([x, y, w, h])

def left_right_eye(seg_mask_eyes):
    cnt, _ = cv2.findContours(seg_mask_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(cnt) < 1:
        return np.asarray([0,0,0,0]), np.asarray([0,0,0,0]) 
    
    if len(cnt) == 2:
        eyes_rects_list = [np.asarray(cv2.boundingRect(c[:,0,:])) for c in cnt]
        eyes_rects = np.array(eyes_rects_list)
        
        eyes_rects = eyes_rects[eyes_rects[:, 0].argsort()]
        return eyes_rects[0], eyes_rects[1]
    
    elif len(cnt) == 1:
        eye_rect = np.asarray(cv2.boundingRect(cnt[0][:, 0, :]))
        return eye_rect, np.asarray([0,0,0,0]) 
    else:
        return np.asarray([0,0,0,0]), np.asarray([0,0,0,0]) 

def calculate_centroids(part_seg_mask):
    contours, _ = cv2.findContours(part_seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centroids = []
    
    if not contours:
        return centroids

    for c in contours:
        M = cv2.moments(c)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centroids.append((cX, cY))
            
    centroids.sort(key=lambda p: p[0])
    
    return centroids


def draw_feature(image, rect, label, color, centroid=None, centroid_color=None):
    if rect is not None and np.any(rect != 0):
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if centroid:
            c_color = centroid_color if centroid_color is not None else color
            cv2.circle(image, centroid, 15, c_color, -1)
            cv2.putText(image, f"{label} C", (centroid[0] + 20, centroid[1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)


def create_proportional_rect(centroid, target_width, aspect_ratio):
    if not centroid: return np.array([0, 0, 0, 0])
    target_height = target_width * aspect_ratio
    new_x = centroid[0] - target_width / 2
    new_y = centroid[1] - target_height / 2
    return np.array([new_x, new_y, target_width, target_height]).astype(int)

def estimate_missing_eye(found_eye_rect, found_eye_centroid, face_rect):
    face_center_x = face_rect[0] + face_rect[2] / 2
    dist_to_center = face_center_x - found_eye_centroid[0]
    estimated_centroid_x = int(face_center_x + dist_to_center)
    estimated_centroid_y = found_eye_centroid[1]
    estimated_centroid = (estimated_centroid_x, estimated_centroid_y)
    
    w, h = found_eye_rect[2], found_eye_rect[3]
    est_x = estimated_centroid_x - w / 2
    est_y = estimated_centroid_y - h / 2
    estimated_rect = np.array([est_x, est_y, w, h]).astype(int)
    
    return estimated_rect, estimated_centroid

def estimate_eyes_from_landmarks(brow_rects, nose_rect):
    left_brow_rect, right_brow_rect = brow_rects
    
    if (left_brow_rect is None and right_brow_rect is None) or nose_rect is None:
        return None, None

    brow_bottom_y_list = []
    if left_brow_rect is not None: brow_bottom_y_list.append(left_brow_rect[1] + left_brow_rect[3])
    if right_brow_rect is not None: brow_bottom_y_list.append(right_brow_rect[1] + right_brow_rect[3])
    
    eye_y_top = int(np.mean(brow_bottom_y_list))
    eye_y_bottom = int(nose_rect[1] + nose_rect[3] / 2)
    
    eye_height = eye_y_bottom - eye_y_top
    if eye_height <= 0: 
        return None, None

    est_left, est_right = None, None

    if left_brow_rect is not None:
        eye_width = left_brow_rect[2]
        eye_center_x = left_brow_rect[0] + left_brow_rect[2] / 2
        eye_x = int(eye_center_x - eye_width / 2)
        
        rect = np.array([eye_x, eye_y_top, eye_width, eye_height])
        centroid = (int(eye_center_x), int(eye_y_top + eye_height / 2))
        est_left = (rect, centroid)

    if right_brow_rect is not None:
        eye_width = right_brow_rect[2]
        eye_center_x = right_brow_rect[0] + right_brow_rect[2] / 2
        eye_x = int(eye_center_x - eye_width / 2)

        rect = np.array([eye_x, eye_y_top, eye_width, eye_height])
        centroid = (int(eye_center_x), int(eye_y_top + eye_height / 2))
        est_right = (rect, centroid)
        
    return est_left, est_right

def adjust_features_by_proportion(face_rect, eye_data, mouth_data, brow_data, nose_data):
    left_eye_rect, right_eye_rect, left_eye_c, right_eye_c = eye_data
    mouth_rect, mouth_c = mouth_data
    left_brow_rect, right_brow_rect, _, _ = brow_data
    nose_rect, _ = nose_data
    
    if not left_eye_c and not right_eye_c:
        est_left, est_right = estimate_eyes_from_landmarks((left_brow_rect, right_brow_rect), nose_rect)
        if est_left: left_eye_rect, left_eye_c = est_left
        if est_right: right_eye_rect, right_eye_c = est_right

    if face_rect is not None and np.any(face_rect != 0):
        if left_eye_c and not right_eye_c:
            right_eye_rect, right_eye_c = estimate_missing_eye(left_eye_rect, left_eye_c, face_rect)
        elif right_eye_c and not left_eye_c:
            left_eye_rect, left_eye_c = estimate_missing_eye(right_eye_rect, right_eye_c, face_rect)

    adj_left_eye, adj_right_eye, adj_mouth = left_eye_rect, right_eye_rect, mouth_rect
    if face_rect is not None and np.any(face_rect != 0):
        face_width = face_rect[2]
        ideal_eye_width = face_width / 5.0
        
        if left_eye_rect is not None and left_eye_c is not None:
            aspect_ratio = left_eye_rect[3] / left_eye_rect[2] if left_eye_rect[2] > 0 else 0.6
            adj_left_eye = create_proportional_rect(left_eye_c, ideal_eye_width, aspect_ratio)

        if right_eye_rect is not None and right_eye_c is not None:
            aspect_ratio = right_eye_rect[3] / right_eye_rect[2] if right_eye_rect[2] > 0 else 0.6
            adj_right_eye = create_proportional_rect(right_eye_c, ideal_eye_width, aspect_ratio)
    
        if left_eye_c and right_eye_c and mouth_rect is not None:
            ideal_mouth_width = abs(right_eye_c[0] - left_eye_c[0])
            aspect_ratio = mouth_rect[3] / mouth_rect[2] if mouth_rect[2] > 0 else 0.4
            adj_mouth = create_proportional_rect(mouth_c, ideal_mouth_width, aspect_ratio)

    return (adj_left_eye, adj_right_eye, adj_mouth), (left_eye_c, right_eye_c, mouth_c)
