import json
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
from tqdm import tqdm


from new_reoptimization import process_single_image
"""
 最佳 Epsilon: 0.9
 Baseline (4k) mAP: 0.3704
 知識引導後 (4k) mAP: 0.3296
 最終真實提升幅度: -0.0408
"""

class TuneConfig:
    GT_ANN_FILE = 'test_IMG/annotations/instances_val2017.json'
    BASELINE_PRED_FILE = 'cv_predictions/baseline_raw_faster_rcnn_full.json'
    KF_MATRIX_FILE = 'kf_matrix_train2017.npy'
    BN_MATRIX_FILE = 'BN_influence.npy'

    EPSILON_CANDIDATES = [0.1, 0.25, 0.5, 0.75, 0.9]
    BK = 5
    LK = 5
    NUM_ITERATIONS = 10


def evaluate_mAP(gt_coco, result_json_path, img_ids=None):
    """只評估指定 img_ids 的 mAP"""
    try:
        dt_coco = gt_coco.loadRes(result_json_path)
        cocoEval = COCOeval(gt_coco, dt_coco, 'bbox')
        if img_ids is not None:
            cocoEval.params.imgIds = img_ids  #  告訴 COCO 只算這些圖片
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats[0]
    except Exception as e:
        print(f" 評估發生錯誤: {e}")
        return 0.0


def run_academic_tuning(config):
    print(" 載入資料庫...")
    coco_gt = COCO(config.GT_ANN_FILE)
    with open(config.BASELINE_PRED_FILE, 'r') as f:
        baseline_preds = json.load(f)

    kf_matrix = np.load(config.KF_MATRIX_FILE)
    bn_matrix = np.load(config.BN_MATRIX_FILE)
    target_matrix = bn_matrix  # 您可以自由切換 kf, bn, 或 kf+bn

    #  學術嚴謹切分：1000 張 Tuning，剩下的 4000 張 Testing
    preds_by_img = defaultdict(list)
    for p in baseline_preds:
        preds_by_img[p['image_id']].append(p)

    unique_img_ids = sorted(list(preds_by_img.keys()))  # 確保每次切分結果固定
    tune_img_ids = unique_img_ids[:1000]
    test_img_ids = unique_img_ids[1000:]

    print(f"\n 資料切分完畢：調參集 {len(tune_img_ids)} 張，測試集 {len(test_img_ids)} 張")

    # 1. 測量 Baseline 在這兩個集合的原始成績
    print("\n--- 基準線評估 (Baseline) ---")
    base_tune_mAP = evaluate_mAP(coco_gt, config.BASELINE_PRED_FILE, tune_img_ids)
    base_test_mAP = evaluate_mAP(coco_gt, config.BASELINE_PRED_FILE, test_img_ids)
    print(f"🌟 Baseline (Tune-1k) mAP: {base_tune_mAP:.4f}")
    print(f"🌟 Baseline (Test-4k) mAP: {base_test_mAP:.4f}")

    # ==========================================
    # 階段一：在 1000 張圖上尋找最佳 Epsilon
    # ==========================================
    print("\n [階段一] 進入 minival-1k 調參場...")
    best_tune_mAP = -1.0
    best_epsilon = None
    os.makedirs('cv_predictions', exist_ok=True)

    for eps in config.EPSILON_CANDIDATES:
        print(f"\n▶ 測試 Epsilon = {eps}")
        temp_preds = []

        for img_id in tqdm(tune_img_ids, desc=f" 優化中", leave=False):
            opt_preds = process_single_image(
                preds_by_img[img_id], target_matrix, config.BK, config.LK, config.NUM_ITERATIONS, eps
            )
            temp_preds.extend(opt_preds)

        temp_out_file = f"cv_predictions/temp_tune_eps_{eps}.json"
        with open(temp_out_file, 'w') as f:
            json.dump(temp_preds, f)

        current_tune_mAP = evaluate_mAP(coco_gt, temp_out_file, tune_img_ids)
        print(f" Epsilon {eps} 在 1k 上的 mAP: {current_tune_mAP:.4f} (差值: {current_tune_mAP - base_tune_mAP:+.4f})")

        if current_tune_mAP > best_tune_mAP:
            best_tune_mAP = current_tune_mAP
            best_epsilon = eps

    print(f"\n [階段一完成] 選定最佳 Epsilon: {best_epsilon}")

    # ==========================================
    # 階段二：拿最佳 Epsilon 去考剩下的 4000 張
    # ==========================================
    print(f"\n [階段二] 拿最佳 Epsilon ({best_epsilon}) 進入 minival-4k 測試場...")
    final_test_preds = []

    for img_id in tqdm(test_img_ids, desc="⚙ 最終測試優化中"):
        opt_preds = process_single_image(
            preds_by_img[img_id], target_matrix, config.BK, config.LK, config.NUM_ITERATIONS, best_epsilon
        )
        final_test_preds.extend(opt_preds)

    final_out_file = f"cv_predictions/final_test_4k_results.json"
    with open(final_out_file, 'w') as f:
        json.dump(final_test_preds, f)

    final_test_mAP = evaluate_mAP(coco_gt, final_out_file, test_img_ids)

    print("\n" + "*" * 15)
    print(" 最終學術報告")
    print(f" 最佳 Epsilon: {best_epsilon}")
    print(f" Baseline (4k) mAP: {base_test_mAP:.4f}")
    print(f" 知識引導後 (4k) mAP: {final_test_mAP:.4f}")
    print(f" 最終真實提升幅度: {(final_test_mAP - base_test_mAP):+.4f}")
    print("*" * 15)


if __name__ == "__main__":
    run_academic_tuning(TuneConfig)