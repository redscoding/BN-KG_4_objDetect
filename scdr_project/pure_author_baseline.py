import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pycocotools.coco import COCO

# 引入專案既有配置
from config import SCDRConfig, COCO_NAMES
from matrix_fusion import build_bn_matrix
from bn_engine import BNEngine
from IFS_author import IntuitionisticFuzzyKnowledgeIntervention
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


def apply_knowledge_fusion(cands, s_matrix, num_iterations=30, epsilon=0.5, k_boxes=5):
    """執行雙軌知識融合：傳統 Fang (Baseline) vs 終極 IFKG (Ours)"""
    if not cands: return [], []

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

    # --- 1. 空間環境矩陣建構 (雙軌共用，完全回歸原始乾淨版本) ---
    iou_matrix = compute_iou_matrix(bboxes, bboxes)
    np.fill_diagonal(iou_matrix, -1.0)

    A_raw = np.zeros((num_boxes, num_boxes))
    if num_boxes > k_boxes:
        top_k_indices = np.argsort(iou_matrix, axis=1)[:, -k_boxes:]
        for i in range(num_boxes):
            A_raw[i, top_k_indices[i]] = 1.0
    else:
        A_raw[iou_matrix >= 0] = 1.0

    col_sums_fang = np.sum(s_matrix, axis=0)
    neighbor_counts_fang = np.sum(A_raw, axis=1)
    denominators_fang = neighbor_counts_fang[:, None] * col_sums_fang[None, :]

    # 完美的基礎知識傳播函數 (不對鄰居做任何模糊加權，確保物理容量純淨)
    def get_context_support(P_current):
        P_neighbors_sum = A_raw @ P_current
        numerators = P_neighbors_sum @ s_matrix
        support = np.zeros_like(numerators)
        mask = denominators_fang > 1e-9
        support[mask] = numerators[mask] / denominators_fang[mask]
        return support

    # ==========================================
    # 軌道 1：傳統 Fang 靜態定錨 (Baseline)
    # ==========================================
    P_fang = P_0.copy()
    for _ in range(num_iterations):
        support_fang = get_context_support(P_fang)
        P_fang = (epsilon * P_0) + ((1.0 - epsilon) * support_fang)

    # ==========================================
    # 軌道 2：IFKG 直覺模糊知識閘門 (Ours)
    # ==========================================
    # 1. 獲取第 0 步純淨的先驗支持度
    S_0 = get_context_support(P_0)

    # 2. 實例化外部引入的閘門引擎，計算每個框獨一無二的動態介入率 G_b
    ifki = IntuitionisticFuzzyKnowledgeIntervention(alpha=1.0)
    G_b = ifki.calculate_gate(P_0, S_0)  # Shape: (num_boxes, 1)

    # 3. 帶有智慧閘門控管的線性迭代
    P_ifis = P_0.copy()
    for _ in range(num_iterations):
        support_ifki = get_context_support(P_ifis)
        # 🌟 終極公式：(1 - G_b) * 視覺特徵 + G_b * 知識推力
        P_ifis = (1.0 - G_b) * P_0 + G_b * support_ifki

    # --- 輸出解析 (完全保持不變，對接後續統計) ---
    output_fang = []
    output_ifis = []
    for i in range(num_boxes):
        sorted_fang = [(IDX_TO_ID[idx], float(score)) for idx, score in enumerate(P_fang[i])]
        sorted_fang.sort(key=lambda x: x[1], reverse=True)
        output_fang.append(int(sorted_fang[0][0]))

        sorted_ifis = [(IDX_TO_ID[idx], float(score)) for idx, score in enumerate(P_ifis[i])]
        sorted_ifis.sort(key=lambda x: x[1], reverse=True)
        output_ifis.append(int(sorted_ifis[0][0]))

    return output_fang, output_ifis


def run_baseline_diagnostics():
    print("\n[System] 啟動基準物理極限與失效模式診斷框架 (IoU-First Mode)...")

    # 初始化矩陣
    bn_engine = BNEngine(SCDRConfig.NETWORK_FILE)
    bn_matrix = build_bn_matrix(bn_engine, VALID_COCO_IDS)
    kf_matrix = np.load(SCDRConfig.KF_MATRIX_FILE)
    hybrid_matrix = (SCDRConfig.HYBRID_WEIGHT * bn_matrix) + ((1.0 - SCDRConfig.HYBRID_WEIGHT) * kf_matrix)

    # 載入標註與預測
    coco_gt = COCO(SCDRConfig.GT_ANN_FILE)
    with open(SCDRConfig.ACTIVE_PRED_FILE, 'r') as f:
        preds = json.load(f)

    img_to_preds = defaultdict(list)
    for p in preds:
        img_to_preds[p['image_id']].append(p)

        # 嚴格統計容器 (擴充為雙軌)
    stats = {
        "total_gt": 0,
        "rank_dist": {"Top-1": 0, "Top-2": 0, "Top-3": 0, "Top-4~5": 0, "Top-6~10": 0, "Out_of_10": 0, "No_Box": 0},
        "recoverability": {"Already_Detected": 0, "Recoverable": 0, "Weak": 0, "Semantic": 0, "No_Box": 0},
        "flips_fang": {"Case_A": 0, "Case_B": 0, "Case_C": 0, "Right_to_Right": 0, "Total_Raw_Flips": 0},
        "flips_ifis": {"Case_A": 0, "Case_B": 0, "Case_C": 0, "Right_to_Right": 0, "Total_Raw_Flips": 0},
        "absolute_defense": 0  # 記錄被 IFIS 成功擋下的 Case B 數量
    }

    IOU_THRESH = 0.5
    SCORE_THRESH = 0.05
    RANK_LIMIT = 5

    print("[System] 執行 1-to-1 貪婪匹配與雙軌對比 (請稍候)...")
    for img_id in tqdm(coco_gt.getImgIds()):
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        cands = img_to_preds.get(img_id, [])

        if not anns: continue

        cand_bboxes = np.array([p.get('bbox', [0, 0, 0, 0]) for p in cands]) if cands else np.empty((0, 4))
        gt_bboxes = np.array([ann['bbox'] for ann in anns])

        # 執行雙軌知識融合
        fusion_top1_fang, fusion_top1_ifis = apply_knowledge_fusion(cands, hybrid_matrix)

        # --- 計算全域無差別擾動 (Total Raw Flips) ---
        for cand_idx, cand in enumerate(cands):
            orig_probs = sorted(cand.get("class_probs_dict", {}).items(), key=lambda x: float(x[1]), reverse=True)
            if not orig_probs: continue
            cnn_top1_cls = int(orig_probs[0][0])

            if cnn_top1_cls != fusion_top1_fang[cand_idx]:
                stats["flips_fang"]["Total_Raw_Flips"] += 1
            if cnn_top1_cls != fusion_top1_ifis[cand_idx]:
                stats["flips_ifis"]["Total_Raw_Flips"] += 1

        # --- 貪婪匹配 (IoU-First) ---
        iou_mat = compute_iou_matrix(gt_bboxes, cand_bboxes)
        used_cand_idx = set()

        for gt_idx, ann in enumerate(anns):
            stats["total_gt"] += 1
            gt_class = ann['category_id']

            best_iou = 0.0
            best_cand_idx = -1

            if iou_mat.size > 0:
                valid_ious = iou_mat[gt_idx].copy()
                for used_idx in used_cand_idx:
                    valid_ious[used_idx] = -1.0

                max_idx = np.argmax(valid_ious)
                if valid_ious[max_idx] >= IOU_THRESH:
                    best_iou = valid_ious[max_idx]
                    best_cand_idx = max_idx

            if best_cand_idx == -1:
                stats["rank_dist"]["No_Box"] += 1
                stats["recoverability"]["No_Box"] += 1
                continue

            used_cand_idx.add(best_cand_idx)
            cand = cands[best_cand_idx]

            orig_probs = sorted(cand.get("class_probs_dict", {}).items(), key=lambda x: float(x[1]), reverse=True)
            if not orig_probs: continue

            cnn_top1_cls = int(orig_probs[0][0])
            gt_score = 0.0
            gt_rank = 999
            for rank_idx, (cls_str, score) in enumerate(orig_probs):
                if int(cls_str) == gt_class:
                    gt_rank = rank_idx + 1
                    gt_score = float(score)
                    break

            # (此處省略 rank_dist 與 recoverability 統計，保留你原本的寫法即可)
            if gt_rank == 1:
                stats["rank_dist"]["Top-1"] += 1
            elif gt_rank == 2:
                stats["rank_dist"]["Top-2"] += 1
            elif gt_rank == 3:
                stats["rank_dist"]["Top-3"] += 1
            elif 4 <= gt_rank <= 5:
                stats["rank_dist"]["Top-4~5"] += 1
            elif 6 <= gt_rank <= 10:
                stats["rank_dist"]["Top-6~10"] += 1
            else:
                stats["rank_dist"]["Out_of_10"] += 1

            if gt_rank == 1 and gt_score > SCORE_THRESH:
                stats["recoverability"]["Already_Detected"] += 1
            elif gt_rank <= RANK_LIMIT and gt_score > SCORE_THRESH:
                stats["recoverability"]["Recoverable"] += 1
            elif gt_rank <= RANK_LIMIT and gt_score <= SCORE_THRESH:
                stats["recoverability"]["Weak"] += 1
            else:
                stats["recoverability"]["Semantic"] += 1

            # ---------------------------------------------------------
            # 統計雙軌 Label Flip
            # ---------------------------------------------------------
            fang_cls = fusion_top1_fang[best_cand_idx]
            ifis_cls = fusion_top1_ifis[best_cand_idx]

            # 評估 Fang
            if cnn_top1_cls != fang_cls:
                if cnn_top1_cls != gt_class and fang_cls == gt_class:
                    stats["flips_fang"]["Case_A"] += 1
                elif cnn_top1_cls == gt_class and fang_cls != gt_class:
                    stats["flips_fang"]["Case_B"] += 1
                else:
                    stats["flips_fang"]["Case_C"] += 1
            else:
                if cnn_top1_cls == gt_class: stats["flips_fang"]["Right_to_Right"] += 1

            # 評估 IFIS
            if cnn_top1_cls != ifis_cls:
                if cnn_top1_cls != gt_class and ifis_cls == gt_class:
                    stats["flips_ifis"]["Case_A"] += 1
                elif cnn_top1_cls == gt_class and ifis_cls != gt_class:
                    stats["flips_ifis"]["Case_B"] += 1
                else:
                    stats["flips_ifis"]["Case_C"] += 1
            else:
                if cnn_top1_cls == gt_class: stats["flips_ifis"]["Right_to_Right"] += 1

            # 絕對防禦指標：Fang 搞砸了，但 IFIS 成功守住
            if cnn_top1_cls == gt_class and fang_cls != gt_class and ifis_cls == gt_class:
                stats["absolute_defense"] += 1



    # ==========================================
    # 產出學術對比戰報
    # ==========================================
    print("\n" + "=" * 70)
    print(" 🎯 動態直覺模糊知識融合 (IFIS) vs 傳統靜態定錨 (Fang) 效能對比")
    print("=" * 70)

    f_stats = stats["flips_fang"]
    i_stats = stats["flips_ifis"]

    fang_nsg = f_stats['Case_A'] - f_stats['Case_B']
    ifis_nsg = i_stats['Case_A'] - i_stats['Case_B']

    print(f"\n【全域擾動分析 (Total Raw Flips)】")
    print(f"  - Fang (靜態): {f_stats['Total_Raw_Flips']:>6} 次盲目干預")
    print(f"  - IFIS (動態): {i_stats['Total_Raw_Flips']:>6} 次控管介入")

    print(f"\n【語意修正對比 (IoU>=0.5 匹配成功)】")
    print(f"  指標{'':<20} | {'Fang (Baseline)':<18} | {'ifki (Ours)':<18}")
    print("-" * 65)
    print(f"  Case A (救援成功)      | {f_stats['Case_A']:<18} | {i_stats['Case_A']:<18}")
    print(f"  Case B (錯誤劣化)      | {f_stats['Case_B']:<18} | {i_stats['Case_B']:<18}")
    print(f"  Case C (錯上加錯)      | {f_stats['Case_C']:<18} | {i_stats['Case_C']:<18}")
    print("-" * 65)
    print(f"  淨語意增益 (NSG)       | {fang_nsg:<18} | {ifis_nsg:<18}")

    print("\n" + "🛡️ " * 3 + "核心防禦戰果" + " 🛡️" * 3)
    print(f"  IFIS 成功阻擋了 {stats['absolute_defense']} 次原本會發生的 Case B 語意災難！")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_baseline_diagnostics()
