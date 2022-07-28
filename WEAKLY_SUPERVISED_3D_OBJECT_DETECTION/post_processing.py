import imp
import numpy as np 
import torch 
from tqdm import tqdm
import wandb
import torchvision
from torchvision import transforms
from collections import Counter
from kitti_utils import get_8points_from_6points

def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    if box1[0] > box2[2] or box1[2] < box2[0] or box1[1] > box2[3] or box1[3] < box2[1]:
        return 0 
    x1_common = max(box1[0], box2[0])
    x2_common = min(box1[2], box2[2])
    y1_common = max(box1[1], box2[1])
    y2_common = min(box1[3], box2[3])

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    common_area = (x2_common - x1_common) * (y2_common - y1_common)
    iou = common_area / (area1 + area2 - common_area)
    return iou


def calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5, inv_class=None, total_cls_num=1):
    AP = []
    class_wise_correct_found = np.zeros((8))
    for class_num in range(total_cls_num):
        # valid_gt_boxes = torch.zeros((0, 6))
        # valid_pred_boxes = torch.zeros((0, 7))

        valid_gt_boxes_ind = torch.where(gt_boxes[:,1] == class_num)
        valid_gt_boxes = gt_boxes[valid_gt_boxes_ind]
        
        valid_pred_boxes_ind = torch.where(pred_boxes[:, 1] == class_num)
        valid_pred_boxes = pred_boxes[valid_pred_boxes_ind]
        
        pred_ind = torch.argsort(valid_pred_boxes[:,2], descending=True)
        valid_pred_boxes = valid_pred_boxes[pred_ind]

        FP = torch.zeros((valid_pred_boxes.shape[0]))
        TP = torch.zeros((valid_pred_boxes.shape[0]))
        total_gts = valid_gt_boxes.shape[0]
        if total_gts == 0:
            print("WHy")
            AP.append(torch.tensor([0]))
            continue
        
        taken_gt_boxes = set()
        for i in range(valid_pred_boxes.shape[0]):
            curr_valid_gt_boxes_ind = torch.where(valid_gt_boxes[:,0] == valid_pred_boxes[i,0])
            curr_valid_gt_boxes = valid_gt_boxes[curr_valid_gt_boxes_ind] 

            best_iou = 0
            for j in range(curr_valid_gt_boxes.shape[0]):
                curr_iou = iou(curr_valid_gt_boxes[j, 2:].reshape(-1), valid_pred_boxes[i, 3:].reshape(-1))
                # curr_iou = torchvision.ops.box_iou(curr_valid_gt_boxes[j, 2:].reshape(-1, 4), 
                #                                    valid_pred_boxes[i, 3:].reshape(-1, 4))
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_gt_idx = j 
            
            if best_iou >= iou_threshold:
                if (best_gt_idx, valid_pred_boxes[i,0]) in taken_gt_boxes:
                    FP[i] = 1 
                else:
                    class_wise_correct_found[class_num] += 1
                    taken_gt_boxes.add((best_gt_idx, valid_pred_boxes[i,0]))
                    TP[i] = 1 
            else:
                FP[i] = 1 

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / total_gts
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        ap = 0
        for i in range(precisions.shape[0]):
            precisions[i] = torch.tensor([max(precisions[i].item(), torch.max(precisions[i:]).item())])
            if i >= 1:
                ap += precisions[i] * (recalls[i] - recalls[i-1]) 

        # precisions = torch.cat((precisions, torch.tensor([0])))
        # recalls = torch.cat((recalls, torch.tensor([1])))
        xs = [recalls[no] for no in range(recalls.shape[0])]
        ys = [precisions[no] for no in range(precisions.shape[0])]
        data = [[recalls[no], precisions[no]] for no in range(precisions.shape[0])]
        table = wandb.Table(data=data, columns=["x", "y"])
        wandb.log({"Precision-Recall Curve ": wandb.plot.line(table, 
                                                            "x", "y", title="P-R curve " + str(iou_threshold))})
        AP.append(ap)
    return AP, xs, ys

def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    valid_threhold = torch.where(confidence_score > threshold)[0]
    bounding_boxes = bounding_boxes[valid_threhold, :]
    confidence_score = confidence_score[valid_threhold]
    boxes = []
    scores = []
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    confidence_score = confidence_score.detach().cpu().numpy()

    while bounding_boxes.shape[0] != 0:
        confidence_max_idx = np.argmax(confidence_score)
        boxes.append(bounding_boxes[confidence_max_idx, :])
        scores.append(confidence_score[confidence_max_idx])
        reference_box = bounding_boxes[confidence_max_idx, :]

        valid_indx = []
        for i in range(bounding_boxes.shape[0]):
            if iou(reference_box, bounding_boxes[i]) < 0.3:
                valid_indx.append(i)
        bounding_boxes = bounding_boxes[valid_indx, :]
        confidence_score = confidence_score[valid_indx]

    boxes = torch.FloatTensor(boxes)
    scores = torch.FloatTensor(scores)
    return boxes, scores



def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=1
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

        # torch.trapz for numerical integration
    return sum(average_precisions) / len(average_precisions)



def intersection_over_union(boxes_preds, boxes_labels, box_format="corners"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image

def plot_3d(gts, proposals, scan):
    proposals = proposals.detach().cpu().numpy()
    gts = gts.numpy()

    plot_proposals = []
    plot_gts = []

    for i in range(proposals.shape[0]):
        proposal = proposals[i].reshape(-1)
        corners = get_8points_from_6points(proposal).transpose((1, 0)).tolist()
        #print(corners, list(zip(*corners)))
        a_dict = {
                    "corners": list(zip(*corners)),
                    "label": "proposal box",
                    "color": [255,0,0],
                }
        plot_proposals.append(a_dict)

    for i in range(gts.shape[0]):
        gt = gts[i].reshape(-1)
        corners = get_8points_from_6points(gt).transpose((1, 0)).tolist()
        a_dict = {
                    "corners": list(zip(*corners)),
                    "label": "gt box",
                    "color": [0,255,0],
                }
        plot_proposals.append(a_dict)

    scan_val = np.where(scan[:, 0] >= 0)
    scan = scan[scan_val]
    scan = scan[:, :3]
    white_color = np.ones_like(scan) * 255
    scan = np.concatenate([scan, white_color], axis=1)
    plot_proposals = np.asarray(plot_proposals)
    wandb.log({"point_clouds_with_bb": wandb.Object3D({
                    "type": "lidar/beta",
                    "points": scan,
                    "boxes": plot_proposals,})
          })


if __name__ == '__main__':
    pass