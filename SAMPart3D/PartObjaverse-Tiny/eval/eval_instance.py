import numpy as np
import json
from os.path import join
from typing import List


def compute_ap(tp, fp, gt_npos, n_bins=100):
    assert len(tp) == len(fp), 'ERROR: the length of true_pos and false_pos is not the same!'

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    rec = tp / gt_npos
    prec = tp / (fp + tp)

    rec = np.insert(rec, 0, 0.0)
    prec = np.insert(prec, 0, 1.0)

    ap = 0.
    delta = 1.0 / n_bins
    
    out_rec = np.arange(0, 1 + delta, delta)
    out_prec = np.zeros((n_bins+1), dtype=np.float32)

    for idx, t in enumerate(out_rec):
        prec1 = prec[rec >= t]
        if len(prec1) == 0:
            p = 0.
        else:
            p = max(prec1)
        
        out_prec[idx] = p
        ap = ap + p / (n_bins + 1)

    return ap


def eval_per_shape_mean_ap(
    part_name_list: List[str],  # The name list of the shape
    pred_sem: np.ndarray,  # Predicted semantic labels, continuous natural numbers, each number is the index of the part_name_list
    pred_ins: np.ndarray,  # Predicted instance labels, continuous natural numbers, each number is the index of the instance (without semantic)
    gt_sem: np.ndarray,  # Ground truth semantic labels
    gt_ins: np.ndarray,  # Ground truth instance labels
    iou_threshold: float = 0.5
    ) -> float:
    
    assert len(pred_sem) == len(pred_sem) == len(gt_sem) == len(gt_ins)

    gt_n_ins = gt_ins.max() + 1
    pred_n_ins = pred_ins.max() + 1
    n_labels = len(part_name_list)

    true_pos_list = [[] for _ in part_name_list]
    false_pos_list = [[] for _ in part_name_list]
    gt_npos = np.zeros((n_labels), dtype=np.int32)

    mapping_insid_to_semid_gt = {}
    for i in range(gt_n_ins):
        sem_id = gt_sem[gt_ins == i][0]
        mapping_insid_to_semid_gt[i] = sem_id
    mapping_insid_to_semid_pred = {}
    for i in range(pred_n_ins):
        sem_id = pred_sem[pred_ins == i][0]
        mapping_insid_to_semid_pred[i] = sem_id

    # classify all gt masks by part categories
    gt_mask_per_cat = [[] for _ in part_name_list]
    for i in range(gt_n_ins):
        sem_id = mapping_insid_to_semid_gt[i]
        gt_mask_per_cat[sem_id].append(i)
        gt_npos[sem_id] += 1
    
    gt_used = np.zeros((gt_n_ins), dtype=np.bool_)

    # enumerate all pred parts
    for idx in range(pred_n_ins):
        sem_id = mapping_insid_to_semid_pred[idx]

        iou_max = 0.0
        cor_gt_id = -1
        for j in gt_mask_per_cat[sem_id]:
            if not gt_used[j]:
                intersect = np.sum((gt_ins == j) & (pred_ins == idx))
                union = np.sum((gt_ins == j) | (pred_ins == idx))
                iou = intersect * 1.0 / union
                
                if iou > iou_max:
                    iou_max = iou
                    cor_gt_id = j
                    
        if iou_max > iou_threshold:
            gt_used[cor_gt_id] = True

            # add in a true positive
            true_pos_list[sem_id].append(True)
            false_pos_list[sem_id].append(False)
        else:
            # add in a false positive
            true_pos_list[sem_id].append(False)
            false_pos_list[sem_id].append(True)

    # compute per-part-category AP
    aps = np.zeros((n_labels), dtype=np.float32)
    ap_valids = np.ones((n_labels), dtype=bool)
    for i in range(n_labels):
        has_pred = (len(true_pos_list[i]) > 0)
        has_gt = (gt_npos[i] > 0)

        if not has_gt:
            ap_valids[i] = False
            continue

        if has_gt and not has_pred:
            continue

        true_pos = np.array(true_pos_list[i], dtype=np.float32)
        false_pos = np.array(false_pos_list[i], dtype=np.float32)

        aps[i] = compute_ap(true_pos, false_pos, gt_npos[i])

    # compute mean AP
    mean_ap = np.sum(aps * ap_valids) / np.sum(ap_valids)

    return aps, ap_valids, gt_npos, mean_ap


def eval_all_shape_mean_ap(meta_path, pred_sem_path, pred_ins_path, gt_sem_path, gt_ins_path, iou_threshold=0.5):
    meta_data = json.load(open(meta_path, 'r'))
    total_mean_ap = []
    categories_list = ["Human-Shape", "Animals", "Daily-Used", "Buildings&&Outdoor", "Transportations", "Plants", "Food", "Electronics"]
    cate_mAP = {}
    for cate in categories_list:
        cate_mAP[cate] = []

    for cate in meta_data.keys():
        for uid in meta_data[cate]:
            print(f"Evaluating {uid}")
            part_name_list = meta_data[cate][uid]
            pred_sem = np.load(join(pred_sem_path, f"{uid}.npy"))
            pred_ins = np.load(join(pred_ins_path, f"{uid}.npy"))
            gt_sem = np.load(join(gt_sem_path, f"{uid}.npy"))
            gt_ins = np.load(join(gt_ins_path, f"{uid}.npy"))

            aps, ap_valids, gt_npos, mean_ap = eval_per_shape_mean_ap(part_name_list, pred_sem, pred_ins, gt_sem, gt_ins, iou_threshold)
            total_mean_ap.append(mean_ap)
            print(f"Mean AP: {mean_ap}")
            with open("eval_ins_results.txt", "a") as f:
                f.write(f"{uid}: {mean_ap}\n")
            cate_mAP[cate].append(mean_ap)

    for cate in categories_list:
        print(f"{cate} Mean AP: {np.mean(cate_mAP[cate])}")
        with open("eval_ins_results.txt", "a") as f:
            f.write(f"{cate} Mean AP: {np.mean(cate_mAP[cate])}\n")
            
    total_mean_ap = np.mean(total_mean_ap)
    print(f"Total Mean AP: {total_mean_ap}")
    with open("eval_ins_results.txt", "a") as f:
        f.write(f"Total Mean AP: {total_mean_ap}\n")


if __name__ == '__main__':
    meta_path = ""
    pred_sem_path = ""
    pred_ins_path = ""
    gt_sem_path = ""
    gt_ins_path = ""
    eval_all_shape_mean_ap(meta_path, pred_sem_path, pred_ins_path, gt_sem_path, gt_ins_path, iou_threshold=0.5)