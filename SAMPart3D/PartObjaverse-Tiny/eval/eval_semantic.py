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


def eval_per_shape_mean_iou(
    part_name_list: List[str],  # The name list of the shape
    pred_sem: np.ndarray,  # Predicted semantic labels, continuous natural numbers, each number is the index of the part_name_list
    gt_sem: np.ndarray,  # Ground truth semantic labels
    ) -> float:

    obj_iou = []
    for i in range(len(part_name_list)):
        if (gt_sem == i).sum() == 0:
            continue
        obj_iou.append(compute_iou(pred_sem == i, gt_sem == i))
    iou = np.mean(obj_iou)

    return iou


def eval_all_shape_mean_iou(meta_path, pred_sem_path, gt_sem_path):

    meta_data = json.load(open(meta_path, 'r'))
    total_miou = []
    categories_list = ["Human-Shape", "Animals", "Daily-Used", "Buildings&&Outdoor", "Transportations", "Plants", "Food", "Electronics"]
    cate_miou = {}
    for cate in categories_list:
        cate_miou[cate] = []

    for cate in meta_data.keys():
        for uid in meta_data[cate]:
            print(f"Evaluating {uid}")
            part_name_list = meta_data[cate][uid]
            pred_sem = np.load(join(pred_sem_path, f"{uid}.npy"))
            gt_sem = np.load(join(gt_sem_path, f"{uid}.npy"))
            obj_iou = eval_per_shape_mean_iou(part_name_list, pred_sem, gt_sem)

            total_miou.append(obj_iou)
            print(f"miou: {obj_iou}")
            with open("eval_sem_results.txt", "a") as f:
                f.write(f"{uid}: {obj_iou}\n")
            cate_miou[cate].append(obj_iou)
    
    for cate in categories_list:
        print(f"{cate} miou: {np.mean(cate_miou[cate])}")
        with open("eval_sem_results.txt", "a") as f:
            f.write(f"{cate} miou: {np.mean(cate_miou[cate])}\n")
            
    total_miou = np.mean(total_miou)
    print(f"Total miou: {total_miou}")
    with open("eval_sem_results.txt", "a") as f:
        f.write(f"Total miou: {total_miou}\n")


if __name__ == '__main__':
    meta_path = ""
    pred_sem_path = ""
    gt_sem_path = ""
    eval_all_shape_mean_iou(meta_path, pred_sem_path, gt_sem_path)