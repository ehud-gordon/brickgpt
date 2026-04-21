import numpy as np
import json
from os.path import join
from typing import List


def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union != 0:
        return (intersection / union) * 100  
    else:
        return 0


def eval_per_shape_part_mean_iou(
    pred_ins: np.ndarray,  # Predicted instance labels, continuous natural numbers, each number is the index of the instance (without semantic)
    gt_ins: np.ndarray,  # Ground truth instance labels
    ) -> float:

    ious = []
    if gt_ins.max() == -1:
        return -1
    for gt_id in np.unique(gt_ins):
        if gt_id == -1:  # Ignore the unassigned group
            continue
        gt_group = gt_ins == gt_id
        best_iou = 0
        for pred_id in np.unique(pred_ins):
            if pred_id == -1:  # Ignore the unassigned group
                continue
            pred_group = pred_ins == pred_id
            iou = compute_iou(pred_group, gt_group)
            if iou > best_iou:
                best_iou = iou
        ious.append(best_iou)
    return np.mean(ious)


def eval_all_shape_part_mean_iou(meta_path, pred_ins_path, gt_ins_path):

    meta_data = json.load(open(meta_path, 'r'))
    total_miou = []
    categories_list = ["Human-Shape", "Animals", "Daily-Used", "Buildings&&Outdoor", "Transportations", "Plants", "Food", "Electronics"]
    cate_miou = {}
    for cate in categories_list:
        cate_miou[cate] = []

    for cate in meta_data.keys():
        for uid in meta_data[cate]:
            print(f"Evaluating {uid}")
            pred_ins = np.load(join(pred_ins_path, f"{uid}.npy"))
            gt_ins = np.load(join(gt_ins_path, f"{uid}.npy"))
            obj_iou = eval_per_shape_part_mean_iou(pred_ins, gt_ins)

            total_miou.append(obj_iou)
            print(f"miou: {obj_iou}")
            with open("eval_partseg_results.txt", "a") as f:
                f.write(f"{uid}: {obj_iou}\n")
            cate_miou[cate].append(obj_iou)
    
    for cate in categories_list:
        print(f"{cate} miou: {np.mean(cate_miou[cate])}")
        with open("eval_partseg_results.txt", "a") as f:
            f.write(f"{cate} miou: {np.mean(cate_miou[cate])}\n")
            
    total_miou = np.mean(total_miou)
    print(f"Total miou: {total_miou}")
    with open("eval_partseg_results.txt", "a") as f:
        f.write(f"Total miou: {total_miou}\n")


if __name__ == '__main__':
    meta_path = ""
    pred_ins_path = ""
    gt_ins_path = ""
    eval_all_shape_part_mean_iou(meta_path, pred_ins_path, gt_ins_path)