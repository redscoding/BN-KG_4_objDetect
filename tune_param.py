import json
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
from tqdm import tqdm

from new_reoptimization import process_single_image


# ==========================================
# 1. 實驗參數設定中心 (高度模組化)
# ==========================================
class TuneConfig:
    GT_ANN_FILE = 'test_IMG/annotations/instances_val2017.json'
    BASELINE_PRED_FILE = 'cv_predictions/baseline_raw_faster_rcnn_full.json'
    KF_MATRIX_FILE = 'kf_matrix_train2017.npy'
    BN_MATRIX_FILE = 'BN_influence.npy'

    # --- 演算法超參數 ---
    BK = 5
    LK = 5
    NUM_ITERATIONS = 10

    # 🌟 模組化開關：是否跳過 1k 調參？
    SKIP_TUNING = True  # 設為 True 即可跳過階段一
    PREDEFINED_EPSILON = 0.9  # 當 SKIP_TUNING=True 時，直接使用這個數值跑 4k
    EPSILON_CANDIDATES = [0.1, 0.25, 0.5, 0.75, 0.9]  # 當 SKIP_TUNING=False 時的搜尋名單


# ==========================================
# 2. 雙指標評估函式 (mAP & Recall)
# ==========================================
def evaluate_metrics(gt_coco, result_json_path, img_ids=None):
    """回傳 (mAP@0.5:0.95, AR@100)"""
    try:
        dt_coco = gt_coco.loadRes(result_json_path)
        cocoEval = COCOeval(gt_coco, dt_coco, 'bbox')
        if img_ids is not None:
            cocoEval.params.imgIds = img_ids

        # 靜音執行，保持終端機乾淨
        from contextlib import redirect_stdout
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

        # 🌟 提取雙指標：stats[0] 是 mAP, stats[8] 是 AR (Recall) @ 100
        return cocoEval.stats[0], cocoEval.stats[8]
    except Exception as e:
        print(f"❌ 評估發生錯誤: {e}")
        return 0.0, 0.0


# ==========================================
# 3. 核心主流程
# ==========================================
def run_academic_tuning(config):
    print("📥 載入資料庫...")
    coco_gt = COCO(config.GT_ANN_FILE)
    with open(config.BASELINE_PRED_FILE, 'r') as f:
        baseline_preds = json.load(f)

    kf_matrix = np.load(config.KF_MATRIX_FILE)
    bn_matrix = np.load(config.BN_MATRIX_FILE)

    # 💡 在這裡切換您要測試的矩陣
    target_matrix = kf_matrix + bn_matrix

    # 切分資料集
    preds_by_img = defaultdict(list)
    for p in baseline_preds:
        preds_by_img[p['image_id']].append(p)

    unique_img_ids = sorted(list(preds_by_img.keys()))
    tune_img_ids = unique_img_ids[:1000]
    test_img_ids = unique_img_ids[1000:]

    # 1. 測量 Baseline 在這兩個集合的原始成績
    print("\n--- 基準線評估 (Baseline) ---")
    base_tune_mAP, base_tune_Recall = evaluate_metrics(coco_gt, config.BASELINE_PRED_FILE, tune_img_ids)
    base_test_mAP, base_test_Recall = evaluate_metrics(coco_gt, config.BASELINE_PRED_FILE, test_img_ids)
    print(f"🌟 Baseline (Tune-1k) -> mAP: {base_tune_mAP:.4f} | Recall: {base_tune_Recall:.4f}")
    print(f"🌟 Baseline (Test-4k) -> mAP: {base_test_mAP:.4f} | Recall: {base_test_Recall:.4f}")

    # ==========================================
    # 階段一：1000 張圖尋找最佳 Epsilon (或跳過)
    # ==========================================
    os.makedirs('cv_predictions', exist_ok=True)
    best_epsilon = None

    if config.SKIP_TUNING:
        print(f"\n⏩ [階段一跳過] 直接指定 Epsilon = {config.PREDEFINED_EPSILON}")
        best_epsilon = config.PREDEFINED_EPSILON
    else:
        print("\n🚀 [階段一] 進入 minival-1k 調參場...")
        best_tune_mAP = -1.0

        for eps in config.EPSILON_CANDIDATES:
            temp_preds = []
            for img_id in tqdm(tune_img_ids, desc=f"   優化中 (eps={eps})", leave=False):
                opt_preds = process_single_image(preds_by_img[img_id], target_matrix, config.BK, config.LK,
                                                 config.NUM_ITERATIONS, eps)
                temp_preds.extend(opt_preds)

            temp_out_file = f"cv_predictions/temp_tune_eps.json"
            with open(temp_out_file, 'w') as f:
                json.dump(temp_preds, f)

            current_tune_mAP, current_tune_Recall = evaluate_metrics(coco_gt, temp_out_file, tune_img_ids)
            print(f"   └─ Epsilon {eps} -> mAP: {current_tune_mAP:.4f} | Recall: {current_tune_Recall:.4f}")

            if current_tune_mAP > best_tune_mAP:
                best_tune_mAP = current_tune_mAP
                best_epsilon = eps

        print(f"\n🎯 [階段一完成] 選定最佳 Epsilon: {best_epsilon}")

    # ==========================================
    # 階段二：拿最佳 Epsilon 去考剩下的 4000 張
    # ==========================================
    print(f"\n🚀 [階段二] 拿 Epsilon ({best_epsilon}) 進入 minival-4k 測試場...")
    final_test_preds = []

    for img_id in tqdm(test_img_ids, desc="⚙️ 最終測試優化中"):
        opt_preds = process_single_image(
            preds_by_img[img_id], target_matrix, config.BK, config.LK, config.NUM_ITERATIONS, best_epsilon
        )
        final_test_preds.extend(opt_preds)

    final_out_file = f"cv_predictions/final_test_4k_results.json"
    with open(final_out_file, 'w') as f:
        json.dump(final_test_preds, f)

    final_test_mAP, final_test_Recall = evaluate_metrics(coco_gt, final_out_file, test_img_ids)

    print("\n" + "🏆" * 15)
    print("📝 最終學術報告 (Test-4k)")
    print(f"💎 鎖定 Epsilon: {best_epsilon}")
    print("-" * 30)
    print(f"📊 Baseline mAP: {base_test_mAP:.4f}")
    print(f"📊 優化後 mAP  : {final_test_mAP:.4f} (提升: {final_test_mAP - base_test_mAP:+.4f})")
    print("-" * 30)
    print(f"📈 Baseline Recall: {base_test_Recall:.4f}")
    print(f"📈 優化後 Recall  : {final_test_Recall:.4f} (提升: {final_test_Recall - base_test_Recall:+.4f})")
    print("🏆" * 15)


if __name__ == "__main__":
    run_academic_tuning(TuneConfig)