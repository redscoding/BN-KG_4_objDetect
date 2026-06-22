import numpy as np

class IntuitionisticFuzzyKnowledgeIntervention:
    def __init__(self, alpha=0.5):
        """
        初始化直覺模糊知識介入引擎 (IFKI) - 尺度校正版
        :param alpha: 猶豫度轉換係數。預設 0.5。
        """
        self.alpha = alpha

    def calculate_gate(self, P_0, S_0):
        """
        計算每個預測框的動態知識介入率 G_b
        :param P_0: CNN 初始機率 (N, C)
        :param S_0: BN 初始支持度 (N, C)
        :return: 知識介入率 G_b 矩陣 (N, 1)
        """
        if P_0.shape[0] == 0:
            return np.array([])

        # 1. 視覺不確定性 (U_CNN)：尺度校正
        # 修正 1：解除 log(80) 的稀疏性封印，改以「局部 5 個類別的混淆」作為最大熵基準
        entropy = -np.sum(P_0 * np.log(P_0 + 1e-9), axis=1)
        max_entropy = np.log(5.0)
        h_norm = np.clip(entropy / max_entropy, 0.0, 1.0)

        # 實施模糊擴張：讓微小的猶豫也能被系統察覺
        h_norm = np.sqrt(h_norm)

        # 2. 知識可靠度 (R_BN)
        s_bn = np.max(S_0, axis=1)

        # 實施模糊擴張：放大 BN 的相對峰值推力 (取代粗暴的 * 2.0)
        s_bn = np.sqrt(s_bn)

        # 3. IFKI 直覺模糊映射 (維持不變)
        mu = h_norm * s_bn
        nu = 1.0 - h_norm

        pi = h_norm * (1.0 - s_bn)
        pi = np.clip(pi, 0.0, 1.0)

        # 4. 解模糊化：計算最終閘門介入率 G_b
        G_b = mu + (self.alpha * pi)

        # 安全截斷並重塑為 (N, 1)
        return np.clip(G_b, 0.0, 1.0).reshape(-1, 1)
