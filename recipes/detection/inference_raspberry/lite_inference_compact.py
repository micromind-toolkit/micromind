"""
Code for inference on a TFLite model for micro controllers.
Adapted from:
https://github.com/AlbertoAncilotto/YoloV8_pose_onnx/blob/master/lite_inference.py

Authors:
    - Alberto Ancilotto, 2023
    - Sebastian Cavada, 2023
"""

import cv2
import time
import colorsys
import numpy as np

try:
    from tensorflow.lite import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter


SCORE_THRESHOLD = 5  # 0.1 if fp32 model, 10 if int8 is used
IMG_SZ = (160, 160)
NUM_CLASSES = 80
COLORS = [
    tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h / 10.0, 1.0, 1.0))
    for h in range(NUM_CLASSES)
]

# Load the TFLite model.
model_path = "best_int8.tflite"
interpreter = Interpreter(model_path=model_path, num_threads=8)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_img(frame):
    """Preprocess the image and convert it from 0-255 range
    to 0-1 range.

    Args:
        frame: The image to be preprocessed.

    Returns:
        The image with the new range.
    """
    img = frame[:, :, ::-1]
    img = img / 255.00
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    return img


def model_inference(input=None):
    # Set input tensor.
    interpreter.set_tensor(input_details[0]["index"], input)

    # Run inference.
    interpreter.invoke()

    # Get output tensor.
    output = interpreter.get_tensor(output_details[0]["index"])
    return output


def post_process(img, output, score_threshold=0.1):
    boxes, confs, classes = non_max_suppression(output, threshold=score_threshold)
    img = plot_objects(img, boxes, confs, classes)
    return img


def non_max_suppression(prediction, class_threshold=0.3, iou_threshold=0.7):
    """Perform the non maximum suppression algorithm on the bounding boxes.
    For every bounding box, only the higher class is kept and the iou is computed
    with the rest of the boxes. If the iou is higher than the threshold, the box is
    deleted.

    Args:
        prediction: The array of RAW predictions made by YOLO.
        threshold: The threshold for the class probability.
        iou_threshold: The threshold for the iou.

    Returns:
        The bounding box that satisfies the constraints.
    """

    bboxes = prediction[:4][:]
    probs = prediction[4:][:]

    keep_idx = np.max(probs, axis=0) > class_threshold

    bboxes = bboxes[:, keep_idx]
    probs = probs[:, keep_idx]

    probs_best_class = np.argmax(probs, axis=0)
    probs_best = np.max(probs, axis=0)

    # If no bounding boxes do nothing
    if len(bboxes) == 0:
        return []

    # Bounding boxes
    x1 = bboxes[0, :] - bboxes[2, :] // 2
    y1 = bboxes[1, :] - bboxes[3, :] // 2
    x2 = bboxes[0, :] + bboxes[2, :] // 2
    y2 = bboxes[1, :] + bboxes[3, :] // 2

    # Compute area of bounding boxes
    areas = (np.abs(x2 - x1) + 1) * (np.abs(y2 - y1) + 1)

    # Indexes sorted by probabilities
    order = probs_best.argsort()[::-1]

    keep = []

    while order.size > 0:

        # Index of current box with the largest probability
        i = order[0]
        keep.append(i)

        # Compute coordinates for intersection with the box with the largest probability
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute width and height of the intersection
        w = np.abs(xx2 - xx1) + 1
        h = np.abs(yy2 - yy1) + 1

        # Compute the ratio of overlap to the area of the intersection
        overlap = (w * h) / (areas[order[1:]] + areas[i] - w * h)

        # Delete all boxes where overlap is larger than the threshold
        inds = np.where(overlap <= iou_threshold)[0]

        order = order[inds + 1]

    return bboxes[:, keep].T, probs_best[keep], probs_best_class[keep]


def plot_objects(img, boxes, confs, classes):
    """Plots the squares around the object and the corresponding
    confidence.

    Args:
        img: The image to plot the squares on.
        boxes: The array of boxes to plot
        confs: The array of confidence relative to the boxes.
        classes: The array of classes relative to the boxes.

    Returns:
        The image with the new data draw on it.
    """

    for i, box in enumerate(boxes):
        cx = int(box[0])
        cy = int(box[1])
        w = int(box[2])
        h = int(box[3])
        bbox = [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]
        cv2.rectangle(
            img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLORS[classes[i]], 2
        )
        cv2.putText(
            img,
            f"conf: {confs[i]:.2f} - cls: {classes[i]}",
            (bbox[0], bbox[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            COLORS[classes[i]],
            2,
            cv2.LINE_AA,
        )

    return img


def calculate_fps(start_time, frames_counter):
    """Calculate Frames Per Second (FPS).

    Args:
        start_time: The time when the frame counting started.
        frames_counter: The number of frames since the start time.

    Returns:
        The calculated FPS.
    """
    elapsed_time = time.time() - start_time
    return frames_counter / elapsed_time


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # Initialize the time and frames counter
    start_time = time.time()
    frames_counter = 0

    while True:
        # Basic operations on the image
        frame = cv2.resize(cap.read()[1], IMG_SZ, interpolation=cv2.INTER_LINEAR)
        input_img = preprocess_img(frame)
        output = model_inference(input_img)
        frame = post_process(frame, output[0], score_threshold=SCORE_THRESHOLD)

        # Increment the frames counter for each frame read
        frames_counter += 1

        # Calculate the actual FPS
        calculated_fps = calculate_fps

        # Display the FPS on the frame
        cv2.putText(
            frame,
            f"FPS : {calculated_fps:.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the image
        cv2.imshow("out", frame)
        cv2.waitKey(1)
