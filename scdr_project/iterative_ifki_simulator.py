# -*- coding: utf-8 -*-
"""
實驗腳本：iterative_experiment.py
功能：消融實驗與極限壓測框架 (Ablation Study on Iteration Steps)
核心方法：Intuitionistic Fuzzy Gated Fixed-Point Iteration (IFG-FPI)
"""

import os
import csv
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 引入專案既有配置與核心引擎
from config import SCDRConfig, COCO_NAMES
from matrix_fusion import build_bn_matrix
from bn_engine import BNEngine
from IFS_author import IntuitionisticFuzzyKnowledgeIntervention
from metrics_tracker import IterativeAblationTracker

# 建立類別映射字典
VALID_COCO_IDS = [k for k in COCO_NAMES.keys() if isinstance(k, int)]
VALID_COCO_IDS.sort()
ID_TO_IDX = {coco_id: idx for idx, coco_id in enumerate(VALID_COCO_IDS)}
IDX_TO_ID = {idx: coco_id for idx, coco_id in enumerate(VALID_COCO_IDS)}


def compute_iou_matrix(bboxes1, bboxes2):
    """向量化計算兩組 BBox 的 IoU 矩陣"""
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))

    x11, y11, w1, h1 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
    x21, y21 = x11 + w1, y11 + h1
    area1 = w1 * h1

    x12, y12, w2, h2 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]
    x22, y22 = x12 + w2, y12 + h2
    area2 = w2 * h2

    xx1 = np.maximum(x11[:, None], x12[None, :])
    yy1 = np.maximum(y11[:, None], y12[None, :])
    xx2 = np.minimum(x21[:, None], x22[None, :])
    yy2 = np.minimum(y21[:, None], y22[None, :])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    iou_matrix = inter / (area1[:, None] + area2[None, :] - inter + 1e-9)
    return iou_matrix


def apply_multi_step_fusion(cands, s_matrix, max_iterations=10, epsilon=0.5, k_boxes=5):
    """執行雙軌多步迭代知識融合，並輸出含 score 的結果供 COCO 使用"""
    if not cands:
        return {}, {}

    num_boxes = len(cands)
    num_classes = s_matrix.shape[0]
    P_0 = np.zeros((num_boxes, num_classes))
    bboxes = np.zeros((num_boxes, 4))

    for i, p in enumerate(cands):
        for k, v in p.get("class_probs_dict", {}).items():
            idx = ID_TO_IDX.get(int(k), -1)
            if idx != -1:
                P_0[i, idx] = float(v)
        bboxes[i] = p.get('bbox', [0, 0, 0, 0])

    iou_matrix = compute_iou_matrix(bboxes, bboxes)
    np.fill_diagonal(iou_matrix, -1.0)

    # Top-K 空間過濾
    A_raw = np.zeros((num_boxes, num_boxes))
    if num_boxes > k_boxes:
        top_k_indices = np.argsort(iou_matrix, axis=1)[:, -k_boxes:]
        for i in range(num_boxes):
            A_raw[i, top_k_indices[i]] = 1.0
    else:
        A_raw[iou_matrix >= 0] = 1.0
    np.fill_diagonal(A_raw, 0.0)

    # Fang 能量最小化 Row-sum 分母
    row_sums_fang = np.sum(s_matrix, axis=1)
    neighbor_counts_fang = np.sum(A_raw, axis=1)
    denominators_fang = neighbor_counts_fang[:, None] * row_sums_fang[None, :]

    def get_context_support(P_current):
        P_neighbors_sum = A_raw @ P_current
        numerators = P_neighbors_sum @ s_matrix
        support = np.zeros_like(numerators)
        mask = denominators_fang > 1e-9
        support[mask] = numerators[mask] / denominators_fang[mask]
        return support

    # 軌道 1：Fang Baseline
    fang_history = {}
    P_fang = P_0.copy()
    for t in range(1, max_iterations + 1):
        support_fang = get_context_support(P_fang)
        P_fang = (epsilon * P_0) + ((1.0 - epsilon) * support_fang)

        step_outputs = []
        for i in range(num_boxes):
            max_idx = np.argmax(P_fang[i])
            step_outputs.append({
                "category_id": int(IDX_TO_ID[max_idx]),
                "score": float(P_fang[i, max_idx])
            })
        fang_history[t] = step_outputs

    # 軌道 2：Ours (IFG-FPI)
    ifis_history = {}
    P_ifis = P_0.copy()
    # 攻擊型設定：alpha=0.5
    ifki = IntuitionisticFuzzyKnowledgeIntervention(alpha=0.5)

    for t in range(1, max_iterations + 1):
        support_ifki = get_context_support(P_ifis)
        G_t_raw = ifki.calculate_gate(P_ifis, support_ifki)
        # 攻擊型設定：允許最高 90% 知識吸收
        G_t = np.clip(G_t_raw, 0.0, 0.9)

        P_ifis = (1.0 - G_t) * P_0 + G_t * support_ifki

        step_outputs = []
        for i in range(num_boxes):
            max_idx = np.argmax(P_ifis[i])
            step_outputs.append({
                "category_id": int(IDX_TO_ID[max_idx]),
                "score": float(P_ifis[i, max_idx])
            })
        ifis_history[t] = step_outputs

    return fang_history, ifis_history


def run_iterative_ablation_study():
    print("\n[System] 啟動多步迭代消融實驗與 COCO 全域指標評估框架...")

    bn_engine = BNEngine(SCDRConfig.NETWORK_FILE)
    bn_matrix = build_bn_matrix(bn_engine, VALID_COCO_IDS)
    kf_matrix = np.load(SCDRConfig.KF_MATRIX_FILE)
    hybrid_matrix = (SCDRConfig.HYBRID_WEIGHT * bn_matrix) + ((1.0 - SCDRConfig.HYBRID_WEIGHT) * kf_matrix)

    coco_gt = COCO(SCDRConfig.GT_ANN_FILE)
    with open(SCDRConfig.ACTIVE_PRED_FILE, 'r') as f:
        preds = json.load(f)

    img_to_preds = defaultdict(list)
    for p in preds:
        img_to_preds[p['image_id']].append(p)

    MAX_ITERATIONS = 10
    IOU_THRESH = 0.5
    # 🌟 這個 tracker 就是負責記錄 T=1 到 T=10 每一輪的 Case A/B
    tracker = IterativeAblationTracker(max_iterations=MAX_ITERATIONS)

    coco_results_fang = []
    coco_results_ours = []

    print(f"[System] 開始單次遍歷影像數據集，執行 1 到 {MAX_ITERATIONS} 步演進壓測...")
    for img_id in tqdm(coco_gt.getImgIds()):
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        cands = img_to_preds.get(img_id, [])

        if not anns: continue

        cand_bboxes = np.array([p.get('bbox', [0, 0, 0, 0]) for p in cands]) if cands else np.empty((0, 4))
        gt_bboxes = np.array([ann['bbox'] for ann in anns])

        fang_history, ifis_history = apply_multi_step_fusion(cands, hybrid_matrix, max_iterations=MAX_ITERATIONS)
        if not fang_history: continue

        # --- 1. 全域擾動統計 (逐圈 T 記錄) ---
        for cand_idx, cand in enumerate(cands):
            orig_probs = sorted(cand.get("class_probs_dict", {}).items(), key=lambda x: float(x[1]), reverse=True)
            if not orig_probs: continue
            cnn_top1_cls = int(orig_probs[0][0])

            # 🌟 這裡就是保留的 T=1~10 逐圈擾動統計
            for t in range(1, MAX_ITERATIONS + 1):
                if cnn_top1_cls != fang_history[t][cand_idx]["category_id"]:
                    tracker.iter_stats[t]["fang"]["Total_Flips"] += 1
                if cnn_top1_cls != ifis_history[t][cand_idx]["category_id"]:
                    tracker.iter_stats[t]["ifis"]["Total_Flips"] += 1

            # 收集最後一輪 (T=10) 供 COCO 使用
            f_final = fang_history[MAX_ITERATIONS][cand_idx]
            i_final = ifis_history[MAX_ITERATIONS][cand_idx]
            coco_results_fang.append({
                "image_id": int(img_id), "category_id": f_final["category_id"],
                "bbox": cand.get('bbox'), "score": f_final["score"]
            })
            coco_results_ours.append({
                "image_id": int(img_id), "category_id": i_final["category_id"],
                "bbox": cand.get('bbox'), "score": i_final["score"]
            })

        # --- 2. 語意翻轉診斷匹配 (逐圈 T 記錄 Case A/B/C) ---
        iou_mat = compute_iou_matrix(gt_bboxes, cand_bboxes)
        used_cand_idx = set()

        for gt_idx, ann in enumerate(anns):
            gt_class = ann['category_id']
            best_cand_idx = -1

            if iou_mat.size > 0:
                valid_ious = iou_mat[gt_idx].copy()
                for used_idx in used_cand_idx: valid_ious[used_idx] = -1.0
                max_idx = np.argmax(valid_ious)
                if valid_ious[max_idx] >= IOU_THRESH: best_cand_idx = max_idx

            if best_cand_idx == -1: continue
            used_cand_idx.add(best_cand_idx)

            cand = cands[best_cand_idx]
            orig_probs = sorted(cand.get("class_probs_dict", {}).items(), key=lambda x: float(x[1]), reverse=True)
            if not orig_probs: continue
            cnn_top1_cls = int(orig_probs[0][0])

            # 🌟 這裡就是保留的 T=1~10 逐圈 Case A, B, C 統計
            for t in range(1, MAX_ITERATIONS + 1):
                fang_cls = fang_history[t][best_cand_idx]["category_id"]
                ifis_cls = ifis_history[t][best_cand_idx]["category_id"]

                # Fang 診斷
                if cnn_top1_cls != fang_cls:
                    if cnn_top1_cls != gt_class and fang_cls == gt_class:
                        tracker.iter_stats[t]["fang"]["Case_A"] += 1
                    elif cnn_top1_cls == gt_class and fang_cls != gt_class:
                        tracker.iter_stats[t]["fang"]["Case_B"] += 1
                    else:
                        tracker.iter_stats[t]["fang"]["Case_C"] += 1

                # IFIS 診斷
                if cnn_top1_cls != ifis_cls:
                    if cnn_top1_cls != gt_class and ifis_cls == gt_class:
                        tracker.iter_stats[t]["ifis"]["Case_A"] += 1
                    elif cnn_top1_cls == gt_class and ifis_cls != gt_class:
                        tracker.iter_stats[t]["ifis"]["Case_B"] += 1
                    else:
                        tracker.iter_stats[t]["ifis"]["Case_C"] += 1

                if cnn_top1_cls == gt_class and fang_cls != gt_class and ifis_cls == gt_class:
                    tracker.iter_stats[t]["absolute_defense"] += 1

            # 紀錄最後一輪的詳細翻盤紀錄 (寫出到 detailed_flips.csv)
            fang_t10 = fang_history[MAX_ITERATIONS][best_cand_idx]["category_id"]
            ifis_t10 = ifis_history[MAX_ITERATIONS][best_cand_idx]["category_id"]
            if cnn_top1_cls != fang_t10 or cnn_top1_cls != ifis_t10:
                tracker.record_flip_detail(img_id, cand.get('bbox'), gt_class, cnn_top1_cls, fang_t10, ifis_t10)

    # =========================================================================
    # 🌟 階段 1：印出你熟悉的「多步迭代消融報表」(T=1~10 Case A/B) 並儲存 CSV
    # =========================================================================
    tracker.print_report()
    tracker.export_csv(out_dir="results")

    # =========================================================================
    # 🏆 階段 2：新增的「COCO 官方標準指標」(AP, AR)
    # =========================================================================
    print("\n" + "═" * 95)
    print(" 🏆 【COCO 官方標準指標對決表 (T=10 Final Iteration)】")
    print("═" * 95)

    # 隱藏 COCOeval 冗長的預設 print (可選)
    import contextlib, sys
    class DummyFile(object):
        def write(self, x): pass

        def flush(self): pass

    print("[COCO Evaluation] 正在評估 Fang (Baseline) 賽道性能...")
    coco_dt_fang = coco_gt.loadRes(coco_results_fang)
    coco_eval_fang = COCOeval(coco_gt, coco_dt_fang, 'bbox')
    with contextlib.redirect_stdout(DummyFile()):  # 隱藏預設輸出
        coco_eval_fang.evaluate()
        coco_eval_fang.accumulate()
        coco_eval_fang.summarize()

    print("[COCO Evaluation] 正在評估 Ours (IFG-FPI) 賽道性能...")
    coco_dt_ours = coco_gt.loadRes(coco_results_ours)
    coco_eval_ours = COCOeval(coco_gt, coco_dt_ours, 'bbox')
    with contextlib.redirect_stdout(DummyFile()):  # 隱藏預設輸出
        coco_eval_ours.evaluate()
        coco_eval_ours.accumulate()
        coco_eval_ours.summarize()

    # 提取核心指標，印出高對齊對決表
    print("\n" + "=" * 80)
    print(" 📊 論文標準指標快速對照矩陣 (Summary Matrix)")
    print("=" * 80)
    print(f"  Metric 指標             |  Fang (Baseline)       |  Ours (IFG-FPI)")
    print("-" * 80)
    print(
        f"  AP [0.5:0.95] (綜合精度) |  {coco_eval_fang.stats[0] * 100:20.4f}%  |  {coco_eval_ours.stats[0] * 100:20.4f}%")
    print(
        f"  AP@50         (寬鬆精度) |  {coco_eval_fang.stats[1] * 100:20.4f}%  |  {coco_eval_ours.stats[1] * 100:20.4f}%")
    print(
        f"  AP@75         (嚴格精度) |  {coco_eval_fang.stats[2] * 100:20.4f}%  |  {coco_eval_ours.stats[2] * 100:20.4f}%")
    print(
        f"  AR@100        (全域召回) |  {coco_eval_fang.stats[11] * 100:20.4f}%  |  {coco_eval_ours.stats[11] * 100:20.4f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_iterative_ablation_study()
