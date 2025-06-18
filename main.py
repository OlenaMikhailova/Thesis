import argparse
import logging
import sys
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont

sys.path.insert(0, str((Path(__file__).parent / "face_parsing").absolute()))

import cv2
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from enum import Enum

from face_parsing.inference import load_model
from face_parsing.utils.common import COLOR_LIST as SEG_MASK_COLORS
from processor import TRANSLATION_DICT, DriverFace, DriverFatigueProcessor, DriverFatigueResult, DriverFatigueRunner, FatigueAnalyzer

logger = logging.getLogger("Driver Fatigue")
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    prog="Driver Fatigue Analyzer",
    usage="python3 main.py",
    description="A simple driver fatigue monitoring script",
)

parser.add_argument("--type", choices=["simple", "withmoments"], required=False, default="simple")


class DisplayMode(Enum):
    NONE = 0
    FACE_MASK = 1
    FACE_ROI = 2


def display_legend(frame: np.typing.NDArray) -> np.typing.NDArray:
    return frame


def blend_mask(
    frame: np.typing.NDArray,
    result: DriverFatigueResult,
) -> np.typing.NDArray:
    if result.mask is None:
        logger.warning("Face mask is None")
        return frame

    mask = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    # Create a color mask
    msk_color = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i, cls_color in enumerate(SEG_MASK_COLORS):
        cls_blob = np.where(mask == i)
        msk_color[cls_blob[0], cls_blob[1], :] = cls_color

    msk_color = msk_color.astype(np.uint8)

    # Blend the image with the segmentation mask
    frame = cv2.addWeighted(frame, 0.6, msk_color, 0.4, 0)
    return frame


def draw_rois(
    frame: np.typing.NDArray,
    result: DriverFatigueResult,
    font: ImageFont.FreeTypeFont,
    panel_width: int = 300
) -> np.typing.NDArray:
    faces: List[DriverFace] = result.faces

    h, w, _ = frame.shape
    display_frame = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    display_frame[0:h, 0:w] = frame
    cv2.rectangle(display_frame, (w, 0), (w + panel_width, h), (20, 20, 20), -1)

    if faces:
        face = faces[0]
        cv2.polylines(display_frame, [face.box_points], True, (0, 255, 0), 2)

        for eye in face.eyes:
            is_closed = (eye.original_label == "Closed")
            color = (0, 0, 255) if is_closed else (0, 255, 255)
            cv2.polylines(display_frame, [eye.box_points], True, color, 2)

        if face.mouth:
            is_fatigue = face.mouth.original_label in ["yawn", "Open"]
            color = (0, 0, 255) if is_fatigue else (255, 255, 0)
            cv2.polylines(display_frame, [face.mouth.box_points], True, color, 2)

    pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    text_x, text_y, line_h = w + 20, 30, 35

    draw.text((text_x, text_y), "СТАТУС ВОДІЯ", font=font, fill=(255,255,255))
    text_y += line_h

    if not faces:
        draw.text((text_x, h//2), "Обличчя не знайдено", font=font, fill=(0,0,255))
    else:
        face = faces[0]
        text_y += line_h
        draw.text((text_x, text_y), "Миттєвий стан:", font=font, fill=(200,200,200))
        text_y += line_h

        if face.eyes:
            for eye in face.eyes:
                name = TRANSLATION_DICT.get(eye.id, eye.id)
                draw.text((text_x, text_y), f"- {name}: {eye.label}", font=font, fill=(255,255,255))
                text_y += line_h
        else:
            draw.text((text_x, text_y), "- Очі не знайдено", font=font, fill=(255,165,0))
            text_y += line_h

        mouth_status = face.mouth.label if face.mouth else "Не визначено"
        draw.text((text_x, text_y), f"- Рот: {mouth_status}", font=font, fill=(255,255,255))
        text_y += line_h * 1.5

        draw.text((text_x, text_y), "Довгостроковий стан:", font=font, fill=(200,200,200))
        text_y += line_h

        hist = TRANSLATION_DICT.get(face.historical_status, face.historical_status)
        color = (255,0,0) if face.historical_status=="PROLONGED_FATIGUE" else (0,255,0)
        draw.text((text_x, text_y), hist, font=font, fill=color)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)



if __name__ == "__main__":
    logger.info("Starting the program")
    logger.info("Pasring CLI arguments")
    args = parser.parse_args()
    logger.info(args.type)

    try:
        frame_seg_model = load_model(
            model_name="resnet34",
            num_classes=19,
            weight_path="prototypes/live_detection_with_image_segmentation/weights/resnet34.pt",
            device=device,
        )
    except ValueError as exc:
        logger.error(exc)
        exit(1)

    analyzer = FatigueAnalyzer(
        fps=24,
        history_seconds=10,
        eye_plateau_sec=2.0,
        yawn_plateau_sec=2.5,
        fatigue_confidence=0.75
    )

    try:
        main_font = ImageFont.truetype("arial.ttf", 16) 
    except OSError:
        print("Не вдалося завантажити шрифт arial.ttf")
        main_font = None

    if args.type == "simple":
        proc = DriverFatigueProcessor(
            frame_seg_model,
            device,
            analyzer,
        )
    elif args.type == "withmoments":
        proc = DriverFatigueProcessor(
            frame_seg_model,
            device,
            analyzer,
        )

    dfr = DriverFatigueRunner(proc).start()

    dm = DisplayMode.NONE
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("No frame was read.")
            break

        frame = cv2.flip(frame, 1)

        dfr.update(frame.copy())
        result = dfr.frame_result
        if not result:
            continue

        if dm == DisplayMode.FACE_MASK:
            logger.debug("DisplayMode: FACE_MASK")
            display = blend_mask(frame, result if result is not None else DriverFatigueResult(mask=np.zeros_like(frame[:, :, 0])))
        elif dm == DisplayMode.FACE_ROI:
            logger.debug("DisplayMode: FACE_ROI")
            display = draw_rois(frame, result, main_font)
        else:
            logger.debug("DisplayMode: NONE")
            display = display_legend(frame)

        cv2.imshow("Driver Monitoring System", display)
        key_pressed = cv2.waitKey(24)
        if key_pressed & 0xFF == ord("q"):
            logger.info("Stopping the program by user")
            break
        elif key_pressed & 0xFF == ord("0"):
            logger.info("Switched to display mode: NONE")
            dm = DisplayMode.NONE
        elif key_pressed & 0xFF == ord("1"):
            logger.info("Switched to display mode: FACE_MASK")
            dm = DisplayMode.FACE_MASK
        elif key_pressed & 0xFF == ord("2"):
            logger.info("Switched to display mode: FACE_ROI")
            dm = DisplayMode.FACE_ROI

    dfr.stop()
    cap.release()
    cv2.destroyAllWindows()

    logging.shutdown()
