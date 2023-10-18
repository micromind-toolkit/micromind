"""
Helper functions.

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023
"""

import numpy as np
import cv2
from collections import defaultdict
import time
import torch
import torchvision


def get_variant_multiples(variant):
    tmp = {
        "n": (0.33, 0.25, 2.0),
        "s": (0.33, 0.50, 2.0),
        "m": (0.67, 0.75, 1.5),
        "l": (1.0, 1.0, 1.0),
        "x": (1, 1.25, 1.0),
    }.get(variant, None)

    return tmp[1], tmp[2], tmp[0]


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Calculate padding value for a convolution operation based on kernel size and dilation.

    This function computes the padding value for a convolution operation to
    maintain the spatial size of the input tensor.
    
    Arguments
    ---------
    k : int
        Kernel size for the convolution operation. If a single integer
        is provided, it's assumed that all dimensions have the same kernel size.
    p : int, optional
        Padding value for the convolution operation. If not provided,
        it will be calculated to maintain the spatial size of the input tensor.
    d : int, optional
        Dilation for the convolution operation. Default is 1.

    Returns
    -------
        The padding value to maintain the spatial size of the input tensor : int
    """
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad

    return p


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points and stride tensors.

    This function generates anchor points for each feature map and stride
    combination. 
    It is commonly used in object detection tasks to define anchor boxes.

    Arguments
    ---------
    feats : torch.Tensor
        A feature map (tensor) from which anchor points will be generated.
    strides : torch.Tensor
        Stride values corresponding to each feature map.
        Strides define the spacing between anchor points.
    grid_cell_offset : float, optional
        Offset to be added to the grid cell coordinates when
        generating anchor points. Default is 0.5.

    Returns
    -------
    anchor_points : torch.Tensor
        Concatenated anchor points for all feature maps as a 2D tensor.
    stride_tensor : torch.Tensor
        Concatenated stride values for all anchor points as a 2D tensor.
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Convert distance predictions to bounding box coordinates.

    This function takes distance predictions and anchor points to calculate
    bounding box coordinates.
    
    Arguments
    ---------
    distance : torch.Tensor
        Tensor containing distance predictions. 
        It should be in the format [lt, rb] if `xywh` is True,
        or [x1y1, x2y2] if `xywh` is False.
    anchor_points : torch.Tensor
        Tensor containing anchor points used for the conversion.
    xywh : bool, optional
        If True, the function returns bounding boxes in the format
        [center_x, center_y, width, height]. 
        If False, it returns bounding boxes in the format [x1, y1, x2, y2].
        Default is True.
    dim : int, optional
        The dimension along which the tensor is split into lt and rb.
        Default is -1.

    Returns
    -------
        Converted bounding box coordinates in the specified format : torch.Tensors
    """
    lt, rb = torch.chunk(distance, chunks=2, dim=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim=1)
    return torch.cat((x1y1, x2y2), dim=1)


def compute_transform(
    image, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32
):
    """Compute a transformation of an image to the specified size and format.

    This function computes a transformation of the input image to the specified
    new size and format, while optionally maintaining the aspect ratio or adding
    padding as needed.
    
    Arguments
    ---------
    image : numpy.ndarray
        The input image to be transformed.
    new_shape : int or tuple, optional
        The target size of the transformed image. If an integer is provided,
        the image is resized to have the same width and height.
        If a tuple of two integers is provided, it represents the new width
        and height. Default is (640, 640).
    auto : bool, optional
        If True, automatically calculates padding to ensure the output size
        is divisible by the specified `stride`. Default is False.
    scaleFill : bool, optional
        If True, scales the image to completely fill the target size without
        maintaining the aspect ratio. Default is False.
    scaleup : bool, optional
        If True, allows the image to be scaled up (enlarged) if necessary.
        Default is True.
    stride : int, optional
        The stride value used for padding calculation when `auto` is True.
        Default is 32.

    Returns
    -------
        The transformed image : numpy.ndarray
    """
    shape = image.shape[:2]  # current shape [height, width]
    new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0) if not scaleup else r
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = (np.mod(dw, stride), np.mod(dh, stride)) if auto else (0.0, 0.0)
    new_unpad = (new_shape[1], new_shape[0]) if scaleFill else new_unpad
    dw /= 2
    dh /= 2
    image = (
        cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        if shape[::-1] != new_unpad
        else image
    )
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return image


def preprocess(im, imgsz=640, model_stride=32, model_pt=True):
    """Preprocess a batch of images for inference.

    This function preprocesses a batch of images for inference by
    resizing, transforming, and normalizing them.
    
    Arguments
    ---------
    im : list of numpy.ndarray or numpy.ndarray
        A batch of input images to be preprocessed. 
        Can be a list of images or a single image as a numpy array.
    imgsz : int, optional
        The target size of the images after preprocessing. 
        Default is 640.
    model_stride : int, optional
        The stride value used for padding calculation when `auto` is True
        in `compute_transform`. Default is 32.
    model_pt : bool, optional
        If True, the function automatically calculates the padding to
        maintain the same shapes for all input images in the batch. 
        Default is True.

    Returns
    -------
    torch.Tensor
        The preprocessed batch of images as a torch.Tensor with shape
        (n, 3, h, w), where n is the number of images, 3 represents the
        RGB channels, and h and w are the height and width of the images.
    """
    same_shapes = all(x.shape == im[0].shape for x in im)
    auto = same_shapes and model_pt
    im = torch.Tensor(
        np.array(
            [
                compute_transform(x, new_shape=imgsz, auto=auto, stride=model_stride)
                for x in im
            ]
        )
    )
    im = torch.stack(im) if im.shape[0] > 1 else im
    im = torch.flip(im, (-1,)).permute(
        0, 3, 1, 2
    )  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


# Post Processing functions
def box_area(box):
    """Calculate the area of bounding boxes.

    This function calculates the area of bounding boxes
    represented as [x1, y1, x2, y2].

    Arguments
    ---------
    box : torch.Tensor
        A tensor containing bounding boxes in the format [x1, y1, x2, y2].

    Returns
    -------
        A tensor containing the area of each bounding box : torch.Tensor
    """
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])


def box_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) between two sets
    of bounding boxes.

    This function computes the IoU between two sets of bounding boxes.

    Arguments
    ---------
    box1 : numpy.ndarray
        The first set of bounding boxes in the format [x1, y1, x2, y2].
    box2 : numpy.ndarray
        The second set of bounding boxes in the format [x1, y1, x2, y2].

    Returns
    -------
    numpy.ndarray
        A 2D numpy array containing the IoU between each pair of bounding
        boxes in box1 and box2.
    """
    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = box_area(box1)[:, None]
    area2 = box_area(box2)[None, :]
    iou = inter / (area1 + area2 - inter)
    return iou


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def postprocess(preds, img, orig_imgs):
    """Perform post-processing on the predictions.

    This function applies post-processing to the predictions,
    including Non-Maximum Suppression (NMS) and scaling of bounding boxes.

    Arguments
    ---------
    preds : list of numpy.ndarray
        A list of prediction arrays from the object detection model.
    img : numpy.ndarray
        The input image on which the predictions were made.
    orig_imgs : numpy.ndarray or list of numpy.ndarray
        The original image(s) before any preprocessing.

    Returns
    -------
    list of numpy.ndarray
        A list of post-processed prediction arrays, each containing bounding
        boxes and associated information.
    """
    print("copying to CPU now for post processing")
    # if you are on CPU, this causes an overflow runtime error. doesn't "seem" to make any difference in the predictions though.
    # TODO: make non_max_suppression in tinygrad - to make this faster
    preds = preds
    preds = non_max_suppression(
        prediction=preds,
        conf_thres=0.25,
        iou_thres=0.7,
        agnostic=False,
        max_det=300,
        multi_label=True,
    )
    all_preds = []
    # TODO: DA SISTEMARE SE IN INGRESO C'è SINGOLA IMMAGINE O BATCH, COSI è BATCH 
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        if not isinstance(orig_imgs, torch.Tensor):
            pred[:, :4] = scale_boxes(tuple(img["img"].shape[2:4]), pred[:, :4], orig_img["ori_shape"][i])
            all_preds.append(pred)
    return all_preds


def draw_bounding_boxes_and_save(
    orig_img_paths, output_img_paths, all_predictions, class_labels, iou_threshold=0.5
):
    """Draw bounding boxes on images based on object detection predictions and
    save the result.

    This function draws bounding boxes on images based on object detection
    predictions and saves the result. It also prints the number of objects
    detected for each class.

    Arguments
    ---------
    orig_img_paths : list of str
        A list of file paths to the original input images.
    output_img_paths : list of str
        A list of file paths to save the images with bounding boxes.
    all_predictions : list of list of numpy.ndarray
        A list of lists of prediction arrays from the object detection model.
    class_labels : list of str
        A list of class labels corresponding to the object classes.
    iou_threshold : float, optional
        The IoU threshold used for non-maximum suppression to remove
        overlapping bounding boxes. Default is 0.5.

    Returns
    -------
        None
    """
    color_dict = {
        label: tuple(
            (((i + 1) * 50) % 256, ((i + 1) * 100) % 256, ((i + 1) * 150) % 256)
        )
        for i, label in enumerate(class_labels)
    }
    font = cv2.FONT_HERSHEY_SIMPLEX

    def is_bright_color(color):
        r, g, b = color
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return brightness > 127

    for img_idx, (orig_img_path, output_img_path, predictions) in enumerate(
        zip(orig_img_paths, output_img_paths, all_predictions)
    ):
        predictions = np.array(predictions)
        orig_img = cv2.imread(orig_img_path)
        height, width, _ = orig_img.shape
        box_thickness = int((height + width) / 400)
        font_scale = (height + width) / 2500

        grouped_preds = defaultdict(list)
        object_count = defaultdict(int)

        for pred_np in predictions:
            grouped_preds[int(pred_np[-1])].append(pred_np)

        def draw_box_and_label(pred, color):
            x1, y1, x2, y2, conf, _ = pred
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, box_thickness)
            label = f"{class_labels[class_id]} {conf:.2f}"
            text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
            label_y, bg_y = (
                (y1 - 4, y1 - text_size[1] - 4)
                if y1 - text_size[1] - 4 > 0
                else (y1 + text_size[1], y1)
            )
            cv2.rectangle(
                orig_img,
                (x1, bg_y),
                (x1 + text_size[0], bg_y + text_size[1]),
                color,
                -1,
            )
            font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
            cv2.putText(
                orig_img,
                label,
                (x1, label_y),
                font,
                font_scale,
                font_color,
                1,
                cv2.LINE_AA,
            )

        for class_id, pred_list in grouped_preds.items():
            pred_list = np.array(pred_list)
            while len(pred_list) > 0:
                max_conf_idx = np.argmax(pred_list[:, 4])
                max_conf_pred = pred_list[max_conf_idx]
                pred_list = np.delete(pred_list, max_conf_idx, axis=0)
                color = color_dict[class_labels[class_id]]
                draw_box_and_label(max_conf_pred, color)
                object_count[class_labels[class_id]] += 1
                iou_scores = box_iou(np.array([max_conf_pred[:4]]), pred_list[:, :4])
                low_iou_indices = np.where(iou_scores[0] < iou_threshold)[0]
                pred_list = pred_list[low_iou_indices]
                for low_conf_pred in pred_list:
                    draw_box_and_label(low_conf_pred, color)

        print(f"Image {img_idx + 1}:")
        print("Objects detected:")
        for obj, count in object_count.items():
            print(f"- {obj}: {count}")

        cv2.imwrite(output_img_path, orig_img)
        print(f"saved detections at {output_img_path}")


def clip_boxes(boxes, shape):
    """Clip bounding boxes to stay within image boundaries.

    This function clips bounding boxes to ensure that they stay within the
    boundaries of the image.

    Arguments
    ---------
    boxes : numpy.ndarray
        An array containing bounding boxes in the format [x1, y1, x2, y2].
    shape : tuple
        A tuple representing the shape of the image in the format (height, width).

    Returns
    -------
        An array containing the clipped bounding boxes : numpy.ndarray
    """
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[0])  # y1, y2
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Scale bounding boxes to match a different image shape.

    This function scales bounding boxes to match a different image
    shape while maintaining their aspect ratio.

    Arguments
    ---------
    img1_shape : tuple
        A tuple representing the shape of the target image in the
        format (height, width).
    boxes : numpy.ndarray or torch.Tensor
        An array or tensor containing bounding boxes in the
        format [x1, y1, x2, y2].
    img0_shape : tuple
        A tuple representing the shape of the source image in the
        format (height, width).
    ratio_pad : float or None, optional
        A scaling factor for the bounding boxes.
        If None, it is calculated based on the aspect ratio of the images.
        Default is None.

    Returns
    -------
        A tensor containing the scaled bounding boxes : torch.Tensor
    """
    gain = (
        ratio_pad
        if ratio_pad
        else min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    )
    pad = (
        (img1_shape[1] - img0_shape[1] * gain) / 2,
        (img1_shape[0] - img0_shape[0] * gain) / 2,
    )
    boxes_np = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
    boxes_np[..., [0, 2]] -= pad[0]
    boxes_np[..., [1, 3]] -= pad[1]
    boxes_np[..., :4] /= gain
    boxes_np = clip_boxes(boxes_np, img0_shape)
    return torch.tensor(boxes_np)


def xywh2xyxy(x):
    """Convert bounding box coordinates from (x, y, width, height)
    to (x1, y1, x2, y2) format.

    This function converts bounding box coordinates from the format
    (center_x, center_y, width, height) to the format (x1, y1, x2, y2),
    where (x1, y1) represents the top-left corner and (x2, y2) represents
    the bottom-right corner of the bounding box.

    Arguments
    ---------
    x : numpy.ndarray or torch.Tensor
        An array or tensor containing bounding box coordinates in the
        format (center_x, center_y, width, height).

    Returns
    -------
    torch.Tensor
        A tensor containing bounding box coordinates in the
        format (x1, y1, x2, y2).
    """
    xy = x[..., :2]  # center x, y
    wh = x[..., 2:4]  # width, height
    xy1 = xy - wh / 2  # top left x, y
    xy2 = xy + wh / 2  # bottom right x, y
    result = np.concatenate((xy1, xy2), axis=-1)
    return torch.Tensor(result)


def label_predictions(all_predictions):
    """Count the number of predictions for each class.

    This function counts the number of predictions for each class
    in a list of prediction arrays.

    Arguments
    ---------
    all_predictions : list of list of numpy.ndarray
        A list of lists of prediction arrays from the object detection model.

    Returns
    -------
    dict
        A dictionary where the keys are class indices and the values are
        the counts of predictions for each class.
    """
    class_index_count = defaultdict(int)
    for predictions in all_predictions:
        predictions = np.array(predictions)
        for pred_np in predictions:
            class_id = int(pred_np[-1])
            class_index_count[class_id] += 1

    return dict(class_index_count)
