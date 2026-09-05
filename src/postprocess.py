"""Model-output agnostic detection postprocessing helpers."""

from __future__ import annotations

import numpy as np


def decode_yolo11_output(
    output: np.ndarray,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    input_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw one-class YOLO11 export and apply class-wise NMS.

    Ultralytics exports this detector with NMS disabled as ``[1, 5, 8400]``:
    four ``xywh`` box values followed by one class score for each candidate.
    The decoder also accepts the transposed ``[1, 8400, 5]`` representation.

    The exported model used by this project emits normalized ``xywh`` values.
    Pass ``input_size=(height, width)`` to convert them to model-input pixel
    coordinates. Leave it as ``None`` when decoding an export that already
    emits pixel coordinates. Letterbox reversal is intentionally separate and
    should be done with :func:`scale_boxes_to_original`.
    """
    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim == 3:
        if predictions.shape[0] != 1:
            raise ValueError("batched output must contain exactly one image")
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError("output must have shape [1, 5, anchors] or [1, anchors, 5]")

    if predictions.shape[0] >= 5 and predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    if predictions.shape[1] < 5:
        raise ValueError("output must contain four box values and at least one class score")

    boxes_xywh = predictions[:, :4]
    if input_size is not None:
        if len(input_size) != 2 or any(int(value) <= 0 for value in input_size):
            raise ValueError("input_size must be a positive (height, width) pair")
        input_height, input_width = (int(value) for value in input_size)
        scale = np.array(
            [input_width, input_height, input_width, input_height],
            dtype=np.float32,
        )
        boxes_xywh = boxes_xywh * scale
    class_scores = predictions[:, 4:]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1).astype(np.int64)
    finite = np.isfinite(boxes_xywh).all(axis=1) & np.isfinite(scores)
    keep = finite & (scores >= confidence_threshold)
    if not np.any(keep):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    boxes_xyxy = xywh_to_xyxy(boxes_xywh[keep])
    kept_indices = classwise_nms(
        boxes_xyxy,
        scores[keep],
        class_ids[keep],
        iou_threshold=iou_threshold,
    )
    return boxes_xyxy[kept_indices], scores[keep][kept_indices], class_ids[keep][kept_indices]


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from center-x, center-y, width, height to corner format."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape[-1] != 4:
        raise ValueError("boxes must have four coordinates")
    x, y, w, h = np.moveaxis(boxes, -1, 0)
    return np.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), axis=-1)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one xyxy box and an array of xyxy boxes."""
    inter_left = np.maximum(box[0], boxes[:, 0])
    inter_top = np.maximum(box[1], boxes[:, 1])
    inter_right = np.minimum(box[2], boxes[:, 2])
    inter_bottom = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(inter_right - inter_left, 0) * np.maximum(inter_bottom - inter_top, 0)
    area_box = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area_boxes = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return inter / np.maximum(area_box + area_boxes - inter, 1e-7)


def classwise_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """Return kept indices after greedy NMS, independently for each class."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    class_ids = np.asarray(class_ids)
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        candidates = np.where(class_ids == class_id)[0]
        order = candidates[np.argsort(scores[candidates])[::-1]]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            overlaps = box_iou(boxes[current], boxes[order[1:]])
            order = order[1:][overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def scale_boxes_to_original(
    boxes: np.ndarray,
    ratio: float,
    padding: tuple[float, float],
    original_shape: tuple[int, int],
) -> np.ndarray:
    """Undo letterbox padding and clip xyxy boxes to the original image."""
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    pad_x, pad_y = padding
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    height, width = original_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes
