# -*- coding: utf-8 -*-
"""
實驗腳本：iterative_experiment.py
功能：消融實驗與極限壓測框架 (Ablation Study on Iteration Steps)
核心方法：Intuitionistic Fuzzy Gated Fixed-Point Iteration (IFG-FPI)

檔案內部依「功能區塊」分組，方便維護與查找：
  [1] IoU 計算工具
  [2] 知識矩陣建構 / 資料載入
  [3] 雙軌融合演算法 (Fang Baseline vs Ours/IFG-FPI)
  [4] Support 體質診斷 (T=1 推力品質分析)
  [5] COCO 評估輔助工具
  [6] 單張圖片處理流程
  [7] 主控流程 (run_iterative_ablation_study)
邏輯與計算方式皆與原始版本保持一致，僅重新排列並加上區塊註解。
"""

import os
import csv
import json
import contextlib
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

MAX_ITERATIONS = 10
IOU_THRESH = 0.5


# ============================================================================
# [1] IoU 計算工具
# ============================================================================

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


# ============================================================================
# [2] 知識矩陣建構 / 資料載入
# ============================================================================

def build_all_matrices():
    """建構 BN 矩陣、載入 KF 矩陣，並組合出 Hybrid 矩陣，最後透過 IFKM 統計進行知識收縮"""
    bn_engine = BNEngine(SCDRConfig.NETWORK_FILE)
    bn_matrix = build_bn_matrix(bn_engine, VALID_COCO_IDS)
    kf_matrix = np.load(SCDRConfig.KF_MATRIX_FILE)

    safe_bn = np.clip(bn_matrix, 0.0, None)
    # 原始有向混合矩陣 S
    hybrid_matrix = kf_matrix + safe_bn * np.log1p(safe_bn)

    # =========================================================================
    # 🌟 新增：載入經驗統計矩陣並計算直覺模糊收縮矩陣 S_hat
    # =========================================================================
    try:
        help_matrix = np.load("priors/help_matrix.npy")
        harm_matrix = np.load("priors/harm_matrix.npy")
        total_matrix = np.load("priors/total_matrix.npy")
        print(
            f"📊 檔案診斷 -> Help最大: {np.max(help_matrix)}, Harm最大: {np.max(harm_matrix)}, Total最大: {np.max(total_matrix)}")

        # 檢查邏輯一致性：理論上 Help + Harm 必須等於 Total
        error_diff = np.max(np.abs((help_matrix + harm_matrix) - total_matrix))
        print(f"⚖️ 數據誤差值 (必須是 0.0): {error_diff}")
        # 1. 初始化全為 0 的矩陣
        mu = np.zeros_like(hybrid_matrix)
        nu = np.zeros_like(hybrid_matrix)

        # 2. 建立安全遮罩：只在 Total > 0 的地方進行除法
        mask = total_matrix > 0
        mu[mask] = help_matrix[mask] / total_matrix[mask]
        nu[mask] = harm_matrix[mask] / total_matrix[mask]

        # 3. 計算可靠度 R (若 Total 為 0，則 mu=0, nu=0，R 預設退回 0.5 中立狀態)
        reliability_matrix = (1.0 + mu - nu) / 2.0

        S_hat = reliability_matrix * hybrid_matrix
        print(
            f"✅ 成功載入 IFKM 先驗統計！可靠度矩陣最大值: {np.max(reliability_matrix):.4f}, 最小值: {np.min(reliability_matrix):.4f}")

    except FileNotFoundError:
        print("⚠️ 警告：找不到 priors/ 資料夾中的統計矩陣！將 Track 4 退回原始 hybrid_matrix。")
        S_hat = hybrid_matrix.copy()

    # 多回傳一個 S_hat 供 Track 4 使用
    return bn_matrix, kf_matrix, hybrid_matrix, S_hat


def load_predictions_grouped_by_image():
    """載入預測檔，並依 image_id 分組成 dict"""
    with open(SCDRConfig.ACTIVE_PRED_FILE, 'r') as f:
        preds = json.load(f)

    img_to_preds = defaultdict(list)
    for p in preds:
        img_to_preds[p['image_id']].append(p)

    return img_to_preds


def apply_multi_step_fusion(cands, s_matrix, S_hat, max_iterations=10, epsilon=0.5, k_boxes=5):
    """執行四軌多步迭代知識融合 (Fang vs Box vs Source vs Edge)"""
    if not cands:
        return {}, {}, {}, {}, None

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

    # =========================================================================
    # 🛑 空間防護罩：Top-K 鄰接矩陣
    # =========================================================================
    A_raw = np.zeros((num_boxes, num_boxes))
    if num_boxes > k_boxes:
        top_k_indices = np.argsort(iou_matrix, axis=1)[:, -k_boxes:]
        for i in range(num_boxes):
            A_raw[i, top_k_indices[i]] = 1.0
    else:
        A_raw[iou_matrix >= 0] = 1.0
    np.fill_diagonal(A_raw, 0.0)

    # Fang 專用的分母計算
    col_sums_fang = np.sum(s_matrix, axis=0)
    neighbor_counts_fang = np.sum(A_raw, axis=1)
    denominators_fang = neighbor_counts_fang[:, None] * col_sums_fang[None, :]

    def get_context_support(P_current):
        P_neighbors_sum = A_raw @ P_current
        numerators = P_neighbors_sum @ s_matrix
        support = np.zeros_like(numerators)
        mask = denominators_fang > 1e-9
        support[mask] = numerators[mask] / denominators_fang[mask]
        return support

    # =========================================================================
    # 🛤️ 軌道 1：Fang Baseline (傳統無差別吸收)
    # =========================================================================
    fang_history = {}
    P_fang = P_0.copy()
    for t in range(1, max_iterations + 1):
        support_fang = get_context_support(P_fang)
        P_fang = (epsilon * P_0) + ((1.0 - epsilon) * support_fang)

        step_outputs = []
        for i in range(num_boxes):
            max_idx = np.argmax(P_fang[i])
            step_outputs.append({"category_id": int(IDX_TO_ID[max_idx]), "score": float(P_fang[i, max_idx])})
        fang_history[t] = step_outputs

    # =========================================================================
    # 🛡️ 軌道 2：Ours (IFG-FPI 舊版 Box-level 接收端防禦)
    # =========================================================================
    ifis_history = {}
    P_ifis = P_0.copy()
    ifki = IntuitionisticFuzzyKnowledgeIntervention(alpha=0.5)

    for t in range(1, max_iterations + 1):
        support_ifki = get_context_support(P_ifis)

        G_t_raw = ifki.calculate_gate(P_ifis, support_ifki)
        G_t = np.clip(G_t_raw, 0.0, 0.9)
        P_ifis = (1.0 - G_t) * P_0 + G_t * support_ifki

        step_outputs = []
        for i in range(num_boxes):
            max_idx = np.argmax(P_ifis[i])
            step_outputs.append({"category_id": int(IDX_TO_ID[max_idx]), "score": float(P_ifis[i, max_idx])})
        ifis_history[t] = step_outputs

    # =========================================================================
    # 🚀 軌道 3：Source-level IFG (發送端淨化 + Exponential IFS)
    # =========================================================================
    source_history = {}
    P_source = P_0.copy()

    for t in range(1, max_iterations + 1):
        P_norm = P_source / (np.sum(P_source, axis=1, keepdims=True) + 1e-9)
        entropy = -np.sum(P_norm * np.log(P_norm + 1e-9), axis=1, keepdims=True)
        max_entropy = np.log(P_source.shape[1])
        H_norm = np.clip(entropy / max_entropy, 0.0, 1.0)

        # 嚴格的指數衰減 IFS (Gamma = 4)
        gamma = 4.0
        mu = np.exp(-gamma * H_norm)
        nu = H_norm
        g = mu * (1.0 - nu)
        g = np.clip(g, 1e-4, 1.0)

        P_weighted = P_norm * g
        weighted_neighbor_sum = A_raw @ P_weighted
        numerators = weighted_neighbor_sum @ s_matrix

        effective_neighbors = A_raw @ g
        row_sum_S = np.sum(s_matrix, axis=1)
        denominators = effective_neighbors * row_sum_S[None, :]

        support_source = numerators / (denominators + 1e-9)
        epsilon_source = 0.5
        P_source = epsilon_source * P_0 + (1.0 - epsilon_source) * support_source

        step_outputs = []
        for i in range(num_boxes):
            max_idx = np.argmax(P_source[i])
            step_outputs.append({"category_id": int(IDX_TO_ID[max_idx]), "score": float(P_source[i, max_idx])})
        source_history[t] = step_outputs

    # =========================================================================
    # 🏆 軌道 4：Instance-wise Dynamic IFKM (保留原變數，升級雙重動態閘門)
    # =========================================================================
    ifkm_history = {}
    P_ifkm = P_0.copy()
    support_ifkm_T1 = None

    # 1. 取得全局 IFKM 矩陣的 Column Sum (用於分母)
    col_sum_S_hat = np.sum(S_hat, axis=0)

    # =========================================================================
    # 🌟 新增模組 A：發送端動態閘門 (Source Entropy Gate)
    # 計算每個 Box 的 Entropy，越不確定的 Box，發出的推力越弱
    # =========================================================================
    # (防呆機制：確保這裡有 normalized_entropy 可以用，如果已經在前面算過可以直接代入)
    num_classes_dynamic = P_0.shape[1]
    entropy = -np.sum(P_0 * np.log(P_0 + 1e-9), axis=1) / np.log(num_classes_dynamic)
    entropy_min = np.min(entropy)
    entropy_max = np.max(entropy)
    if entropy_max - entropy_min > 1e-9:
        normalized_entropy = (entropy - entropy_min) / (entropy_max - entropy_min)
    else:
        normalized_entropy = np.zeros_like(entropy)

    source_gate = 1.0 - normalized_entropy  # 信心越高 (Entropy低)，Gate 越接近 1

    # =========================================================================
    # 🌟 新增模組 B：接收端動態閘門 (Target Plausibility Gate)
    # 每個 Box 只有 P0 排名前 5 的類別，才允許接收推力 (阻斷無中生有的毒藥)
    # =========================================================================
    top_k_targets = 5
    target_gate = np.zeros_like(P_0)
    for i in range(num_boxes):
        top_indices = np.argsort(P_0[i])[-top_k_targets:]
        target_gate[i, top_indices] = 1.0

    # 準備分母的空間部分
    neighbor_counts = np.sum(A_raw, axis=1)

    # =========================================================================
    # 🔄 迭代推論開始
    # =========================================================================
    for t in range(1, max_iterations + 1):

        # 【1】套用發送端閘門：把 Box 的發言權重乘上目前的機率
        P_gated = P_ifkm * source_gate[:, None]

        # 【2】空間傳遞：鄰居把審查過後的機率傳過來
        P_neighbors_sum = A_raw @ P_gated

        # 【3】套用歷史解毒矩陣 (Global IFKM)
        numerators_raw = P_neighbors_sum @ S_hat

        # 【4】套用接收端閘門：把根本不可能的類別推力強制歸零
        numerators_ifkm = numerators_raw * target_gate

        # 計算分母與 Support (同步用 target_gate 過濾分母，避免機率被過度稀釋)
        denominators_ifkm = neighbor_counts[:, None] * col_sum_S_hat[None, :]
        denominators_ifkm = denominators_ifkm * target_gate

        # 安全相除
        support_ifkm = np.zeros_like(numerators_ifkm)
        mask = denominators_ifkm > 1e-9
        support_ifkm[mask] = numerators_ifkm[mask] / denominators_ifkm[mask]

        # 紀錄第一圈的推力供後續分析
        if t == 1:
            support_ifkm_T1 = support_ifkm.copy()

        # 權重步進更新
        epsilon_k = 0.5
        P_ifkm = epsilon_k * P_0 + (1.0 - epsilon_k) * support_ifkm

        # 紀錄當前迭代步的預測結果
        step_outputs = []
        for i in range(num_boxes):
            max_idx = np.argmax(P_ifkm[i])
            step_outputs.append({
                "category_id": int(IDX_TO_ID[max_idx]),
                "score": float(P_ifkm[i, max_idx])
            })
        ifkm_history[t] = step_outputs

    # 🌟 完美回傳：四軌齊發
    return fang_history, ifis_history, source_history, ifkm_history, support_ifkm_T1#切換T=1 knowledge support分析


# ============================================================================
# [4] Support 體質診斷 (T=1 推力品質分析)
# ============================================================================

def new_support_diag():
    """建立一份全新的 Support 體質追蹤字典"""
    return {
        "gt_rank_1": 0,
        "gt_rank_2_5": 0,
        "gt_rank_6_10": 0,
        "gt_rank_out": 0,
        "rescuable_total": 0,
        "rescuable_favors_gt": 0,
        "rescuable_fails": 0,
        "avg_support_diff": [],  # 用來算 Support[GT] - Support[Top1]
        # 👇 細分 [情境 1] 知識有用的推力強度
        "diff_strong": 0,  # diff >= 0.15 (強推力：翻盤主力)
        "diff_medium": 0,  # 0.05 <= diff < 0.15 (中推力：有機會翻盤)
        "diff_weak": 0,  # diff < 0.05 (弱推力：方向對但杯水車薪)
    }


def analyze_t1_support_quality(support_diag, support_T1, best_cand_idx,
                               gt_class, cnn_top1_cls, orig_probs):
    """
    🕵️‍♂️ 【診斷核心】分析 T=1 的 Knowledge Support 體質
    統計 GT 在 Support 向量中的排名，並分析「可救樣本」的推力方向與強度。
    """
    # 注意：GT 的 class ID 必須轉換成矩陣的 Index
    gt_idx = ID_TO_IDX.get(gt_class, -1)
    cnn_top1_idx = ID_TO_IDX.get(cnn_top1_cls, -1)

    if gt_idx == -1 or cnn_top1_idx == -1:
        return

    support_vec = support_T1[best_cand_idx]

    # 1. 統計 GT 在 Support 向量中的排名
    # argsort 是從小排到大，[::-1] 反轉變成從大排到小
    sorted_support_indices = np.argsort(support_vec)[::-1]
    gt_rank_in_support = np.where(sorted_support_indices == gt_idx)[0][0] + 1

    if gt_rank_in_support == 1:
        support_diag["gt_rank_1"] += 1
    elif gt_rank_in_support <= 5:
        support_diag["gt_rank_2_5"] += 1
    elif gt_rank_in_support <= 10:
        support_diag["gt_rank_6_10"] += 1
    else:
        support_diag["gt_rank_out"] += 1

    # 2. 分析「可救樣本 (Rescuable Samples)」
    # 找出 GT 在 CNN 原本的排名與分數
    cnn_gt_rank = -1
    cnn_gt_score = 0.0
    for r, (cls_str, score_str) in enumerate(orig_probs):
        if int(cls_str) == gt_class:
            cnn_gt_rank = r + 1
            cnn_gt_score = float(score_str)
            break

    # 條件：CNN 原本猜錯，但 GT 在 Top-5 內且分數 > 0.05
    if cnn_top1_cls != gt_class and (2 <= cnn_gt_rank <= 5) and cnn_gt_score > 0.05:
        support_diag["rescuable_total"] += 1

        # 比較推力：Support 給 GT 的推力 vs 給錯誤 Top1 的推力
        supp_gt = support_vec[gt_idx]
        supp_top1 = support_vec[cnn_top1_idx]
        diff = supp_gt - supp_top1

        support_diag["avg_support_diff"].append(diff)

        if diff > 0:
            support_diag["rescuable_favors_gt"] += 1
            if diff >= 0.15:
                support_diag["diff_strong"] += 1
            elif diff >= 0.05:
                support_diag["diff_medium"] += 1
            else:
                support_diag["diff_weak"] += 1
        else:
            support_diag["rescuable_fails"] += 1


def print_support_diagnostics_report(support_diag):
    """印出 Support 知識推力深度體質分析報告"""
    total_matched_gt = (support_diag["gt_rank_1"] + support_diag["gt_rank_2_5"]
                        + support_diag["gt_rank_6_10"] + support_diag["gt_rank_out"])

    print("\n" + "═" * 70)
    print(" 🕵️‍♂️ 【深度診斷】 T=1 Knowledge Support 體質分析報告")
    print("═" * 70)
    if total_matched_gt > 0:
        print(f" ➤ 成功匹配真實物件 (Matched GTs) : {total_matched_gt} 個")
        print(
            f" ➤ Support 排名 Top 1             : {support_diag['gt_rank_1']} ({support_diag['gt_rank_1'] / total_matched_gt:.1%})")
        print(
            f" ➤ Support 排名 Top 2~5           : {support_diag['gt_rank_2_5']} ({support_diag['gt_rank_2_5'] / total_matched_gt:.1%})")
        print(
            f" ➤ Support 排名 Top 6~10          : {support_diag['gt_rank_6_10']} ({support_diag['gt_rank_6_10'] / total_matched_gt:.1%})")
        print(
            f" ➤ Support 排名 10 名外 (沒救了)    : {support_diag['gt_rank_out']} ({support_diag['gt_rank_out'] / total_matched_gt:.1%})")

    print("\n 🎯 【可救樣本分析】 (定義：CNN 原猜錯，但 GT 在 Top-5 且 Score > 0.05)")
    print("-" * 70)
    print(f" 總計篩選出可救樣本 : {support_diag['rescuable_total']} 個")

    if support_diag['rescuable_total'] > 0:
        favors = support_diag['rescuable_favors_gt']
        fails = support_diag['rescuable_fails']
        avg_diff = np.mean(support_diag['avg_support_diff'])

        print(
            f"  [情境 1] 知識有用 (Support[GT] > Support[Top1]) : {favors} 個 ({favors / support_diag['rescuable_total']:.1%})")
        # 👇 印出推力強度的階層分析
        if favors > 0:
            print(f"      ↳ [強推力] 差距 >= 0.15 (絕對翻盤)    : {support_diag['diff_strong']} 個")
            print(f"      ↳ [中推力] 0.05 <= 差距 < 0.15        : {support_diag['diff_medium']} 個")
            print(f"      ↳ [弱推力] 差距 < 0.05  (杯水車薪)    : {support_diag['diff_weak']} 個")
        print(
            f"  [情境 2] 知識有害 (Support[Top1] > Support[GT]) : {fails} 個 ({fails / support_diag['rescuable_total']:.1%})")
        print(f"  [推力差] 平均推力差距 (Diff)                      : {avg_diff:.6f}")

        if avg_diff > 0:
            print("  💡 結論：知識矩陣方向正確，但推力太弱。建議研究「動態分數轉換 (Score Boost)」。")
        else:
            print("  ⚠️ 結論：知識矩陣反向扯後腿！Support 把錯誤的 Top1 推得更高。需重新檢視 S 矩陣！")
    print("═" * 70 + "\n")


# ============================================================================
# [5] COCO 評估輔助工具
# ============================================================================

class DummyFile(object):
    """用於吃掉 COCOeval 內建的冗長 print 輸出"""

    def write(self, x):
        pass

    def flush(self):
        pass


def run_coco_eval_silently(coco_gt, coco_results, track_label=""):
    """載入預測結果並執行 COCOeval（evaluate -> accumulate -> summarize），抑制冗長輸出"""
    if track_label:
        print(f"[COCO Evaluation] 正在評估 {track_label} 賽道性能...")

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    with contextlib.redirect_stdout(DummyFile()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return coco_eval


def print_summary_matrix(coco_eval_fang, coco_eval_ours, coco_eval_source, coco_eval_ifkm):
    """印出論文標準指標快速對照矩陣 (拆分為兩張易讀的子表格)"""

    # =========================================================================
    # 🏆 表格一：傳統 Baseline vs 舊版 Box-level 接收端防禦
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 🏆 官方標準指標對照 (Part 1: 傳統 Baseline vs Box-level 防禦)")
    print("=" * 80)
    print(f"  Metric 指標             |  Fang (Baseline)       |  Ours (Box-level)")
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

    # =========================================================================
    # 🏆 表格二：全新 Source-level 發送端淨化 vs 終極 Edge-level 連線審查
    # =========================================================================
    print("=" * 80)
    print(" 🏆 官方標準指標對照 (Part 2: Source-level 淨化 vs 終極 IFKM-level)")
    print("=" * 80)
    print(f"  Metric 指標             |  Ours (Source-level)   |  Ours (IFKM-level)")
    print("-" * 80)
    print(
        f"  AP [0.5:0.95] (綜合精度) |  {coco_eval_source.stats[0] * 100:20.4f}%  |  {coco_eval_ifkm.stats[0] * 100:20.4f}%")
    print(
        f"  AP@50         (寬鬆精度) |  {coco_eval_source.stats[1] * 100:20.4f}%  |  {coco_eval_ifkm.stats[1] * 100:20.4f}%")
    print(
        f"  AP@75         (嚴格精度) |  {coco_eval_source.stats[2] * 100:20.4f}%  |  {coco_eval_ifkm.stats[2] * 100:20.4f}%")
    print(
        f"  AR@100        (全域召回) |  {coco_eval_source.stats[11] * 100:20.4f}%  |  {coco_eval_ifkm.stats[11] * 100:20.4f}%")
    print("=" * 80 + "\n")


# ============================================================================
# [6] 單張圖片處理流程
# ============================================================================

def _process_single_image(img_id, anns, cands, hybrid_matrix, S_hat, tracker, support_diag,
                          coco_results_fang, coco_results_ours, coco_results_source, coco_results_ifkm): # 🌟 加在這裡

    """處理單張圖片：執行雙軌融合、全域擾動統計、語意翻轉診斷匹配"""
    cand_bboxes = np.array([p.get('bbox', [0, 0, 0, 0]) for p in cands]) if cands else np.empty((0, 4))
    gt_bboxes = np.array([ann['bbox'] for ann in anns])

    fang_history, ifis_history, source_history,ifkm_history, support_T1 = apply_multi_step_fusion(
        cands, hybrid_matrix, S_hat, max_iterations=MAX_ITERATIONS
    )
    if not fang_history:
        return

    # --- 1. 全域擾動統計 (逐圈 T 記錄) ---
    for cand_idx, cand in enumerate(cands):
        orig_probs = sorted(cand.get("class_probs_dict", {}).items(), key=lambda x: float(x[1]), reverse=True)
        if not orig_probs:
            continue
        cnn_top1_cls = int(orig_probs[0][0])

        # 🌟 這裡就是保留的 T=1~10 逐圈擾動統計
        for t in range(1, MAX_ITERATIONS + 1):
            if cnn_top1_cls != fang_history[t][cand_idx]["category_id"]:
                tracker.iter_stats[t]["fang"]["Total_Flips"] += 1
            if cnn_top1_cls != ifis_history[t][cand_idx]["category_id"]:
                tracker.iter_stats[t]["ifis"]["Total_Flips"] += 1
            if cnn_top1_cls != source_history[t][cand_idx]["category_id"]:
                tracker.iter_stats[t]["source"]["Total_Flips"] += 1
            if cnn_top1_cls != ifkm_history[t][cand_idx]["category_id"]:
                tracker.iter_stats[t]["ifkm"]["Total_Flips"] += 1

        # 收集最後一輪 (T=10) 供 COCO 使用
        f_final = fang_history[MAX_ITERATIONS][cand_idx]
        i_final = ifis_history[MAX_ITERATIONS][cand_idx]
        s_final = source_history[MAX_ITERATIONS][cand_idx]
        e_final = ifkm_history[MAX_ITERATIONS][cand_idx]

        coco_results_fang.append({
            "image_id": int(img_id), "category_id": f_final["category_id"],
            "bbox": cand.get('bbox'), "score": f_final["score"]
        })
        coco_results_ours.append({
            "image_id": int(img_id), "category_id": i_final["category_id"],
            "bbox": cand.get('bbox'), "score": i_final["score"]
        })
        coco_results_source.append({
            "image_id": int(img_id), "category_id": s_final["category_id"],
            "bbox": cand.get('bbox'), "score": s_final["score"]
        })
        coco_results_ifkm.append({
            "image_id": int(img_id), "category_id": e_final["category_id"],
            "bbox": cand.get('bbox'), "score": e_final["score"]
        })

    # --- 2. 語意翻轉診斷匹配 (逐圈 T 記錄 Case A/B/C) ---
    iou_mat = compute_iou_matrix(gt_bboxes, cand_bboxes)
    used_cand_idx = set()

    for gt_idx, ann in enumerate(anns):
        gt_class = ann['category_id']
        best_cand_idx = -1

        if iou_mat.size > 0:
            valid_ious = iou_mat[gt_idx].copy()
            for used_idx in used_cand_idx:
                valid_ious[used_idx] = -1.0
            max_idx = np.argmax(valid_ious)
            if valid_ious[max_idx] >= IOU_THRESH:
                best_cand_idx = max_idx

        if best_cand_idx == -1:
            continue
        used_cand_idx.add(best_cand_idx)

        cand = cands[best_cand_idx]
        orig_probs = sorted(cand.get("class_probs_dict", {}).items(), key=lambda x: float(x[1]), reverse=True)
        if not orig_probs:
            continue
        cnn_top1_cls = int(orig_probs[0][0])

        # 🕵️‍♂️ 【診斷核心】分析 T=1 的 Knowledge Support 體質
        analyze_t1_support_quality(
            support_diag, support_T1, best_cand_idx,
            gt_class, cnn_top1_cls, orig_probs
        )

        # 🌟 這裡就是保留的 T=1~10 逐圈 Case A, B, C 統計
        for t in range(1, MAX_ITERATIONS + 1):
            fang_cls = fang_history[t][best_cand_idx]["category_id"]
            ifis_cls = ifis_history[t][best_cand_idx]["category_id"]
            source_cls = source_history[t][best_cand_idx]["category_id"]
            ifkm_cls = ifkm_history[t][best_cand_idx]["category_id"]

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

            if cnn_top1_cls != source_cls:
                if cnn_top1_cls != gt_class and source_cls == gt_class:
                    tracker.iter_stats[t]["source"]["Case_A"] += 1
                elif cnn_top1_cls == gt_class and source_cls != gt_class:
                    tracker.iter_stats[t]["source"]["Case_B"] += 1
                else:
                    tracker.iter_stats[t]["source"]["Case_C"] += 1

            # --- Edge-level 統計 (🌟 新增) ---
            if cnn_top1_cls != ifkm_cls:
                if cnn_top1_cls != gt_class and ifkm_cls == gt_class:
                    tracker.iter_stats[t]["ifkm"]["Case_A"] += 1
                elif cnn_top1_cls == gt_class and ifkm_cls != gt_class:
                    tracker.iter_stats[t]["ifkm"]["Case_B"] += 1
                else:
                    tracker.iter_stats[t]["ifkm"]["Case_C"] += 1

            if cnn_top1_cls == gt_class and fang_cls != gt_class and ifis_cls == gt_class:
                tracker.iter_stats[t]["ifis"]["absolute_defense"] += 1
            if cnn_top1_cls == gt_class and fang_cls != gt_class and source_cls == gt_class:
                tracker.iter_stats[t]["source"]["absolute_defense"] += 1
            if cnn_top1_cls == gt_class and fang_cls != gt_class and ifkm_cls == gt_class:
                tracker.iter_stats[t]["ifkm"]["absolute_defense"] += 1

        # 紀錄最後一輪的詳細翻盤紀錄 (寫出到 detailed_flips.csv)
        fang_t10 = fang_history[MAX_ITERATIONS][best_cand_idx]["category_id"]
        ifis_t10 = ifis_history[MAX_ITERATIONS][best_cand_idx]["category_id"]
        if cnn_top1_cls != fang_t10 or cnn_top1_cls != ifis_t10:
            tracker.record_flip_detail(img_id, cand.get('bbox'), gt_class, cnn_top1_cls, fang_t10, ifis_t10)


# ============================================================================
# [7] 主控流程
# ============================================================================

def run_iterative_ablation_study():
    print("\n[System] 啟動多步迭代消融實驗與 COCO 全域指標評估框架...")

    _bn_matrix, _kf_matrix, hybrid_matrix, S_hat = build_all_matrices()

    coco_gt = COCO(SCDRConfig.GT_ANN_FILE)
    img_to_preds = load_predictions_grouped_by_image()

    # 🌟 Support 知識推力體質分析追蹤器
    support_diag = new_support_diag()

    # 🌟 這個 tracker 就是負責記錄 T=1 到 T=10 每一輪的 Case A/B
    tracker = IterativeAblationTracker(max_iterations=MAX_ITERATIONS)

    coco_results_fang = []
    coco_results_ours = []
    coco_results_source = []
    coco_results_ifkm = []

    print(f"[System] 開始單次遍歷影像數據集，執行 1 到 {MAX_ITERATIONS} 步演進壓測...")
    for img_id in tqdm(coco_gt.getImgIds()):
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        cands = img_to_preds.get(img_id, [])

        if not anns or not cands:
            continue

        _process_single_image(
            img_id, anns, cands, hybrid_matrix, S_hat, tracker, support_diag,
            coco_results_fang, coco_results_ours, coco_results_source, coco_results_ifkm
        )

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

    coco_eval_fang = run_coco_eval_silently(coco_gt, coco_results_fang, track_label="Fang (Baseline)")

    # =========================================================================
    # 🕵️‍♂️ 印出 Support 知識推力深度體質分析報告
    # =========================================================================
    print_support_diagnostics_report(support_diag)

    coco_eval_ours = run_coco_eval_silently(coco_gt, coco_results_ours, track_label="Ours (IFG-FPI)")
    coco_eval_source = run_coco_eval_silently(coco_gt, coco_results_source, track_label="Ours (Source-level IFG)")
    coco_eval_ifkm = run_coco_eval_silently(coco_gt, coco_results_ifkm,track_label="Ours (IFKM IFG)")

    # 提取核心指標，印出高對齊對決表
    print_summary_matrix(coco_eval_fang, coco_eval_ours, coco_eval_source, coco_eval_ifkm)


if __name__ == "__main__":
    run_iterative_ablation_study()
