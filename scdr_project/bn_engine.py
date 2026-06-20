# bn_engine.py
import pysmile
import numpy as np
from config import COCO_NAMES
import pysmile_license

#v1
class BNEngine:
    def __init__(self, network_file):
        print(f"⏳ 載入貝氏網路: {network_file}")
        self.net = pysmile.Network()
        self.net.read_file(network_file)
        self.net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.EPIS_SAMPLING)
        self.net.set_sample_count(10000)

        # 快取節點狀態與 Prior
        self.positive_state_idx = {}
        self.prior_cache = {}
        self._init_cache()

    def _init_cache(self):
        for cat_name in COCO_NAMES.values():
            try:
                outcomes = self.net.get_outcome_ids(cat_name)
                if 'True' in outcomes:
                    self.positive_state_idx[cat_name] = outcomes.index('True')
                elif 'Present' in outcomes:
                    self.positive_state_idx[cat_name] = outcomes.index('Present')
                else:
                    self.positive_state_idx[cat_name] = 1
            except:
                self.positive_state_idx[cat_name] = 1

        self.net.update_beliefs()
        for cat_name in COCO_NAMES.values():
            try:
                self.prior_cache[cat_name] = float(self.net.get_node_value(cat_name)[self.positive_state_idx[cat_name]])
            except:
                self.prior_cache[cat_name] = 0.0

    def set_evidence(self, evidence_nodes):
        self.net.clear_all_evidence()
        if not evidence_nodes: return

        for e_cat, soft_score in evidence_nodes.items():
            if e_cat not in COCO_NAMES: continue
            c_name = COCO_NAMES[e_cat]
            idx = self.positive_state_idx.get(c_name, 1)
            try:
                self.net.set_virtual_evidence(
                    c_name,
                    [1.0 - soft_score, soft_score] if idx == 1 else [soft_score, 1.0 - soft_score]
                )
            except:
                pass

        try:
            self.net.update_beliefs()
        except:
            pass

    def get_posterior(self, coco_id):
        if coco_id not in COCO_NAMES: return 0.0
        cat_name = COCO_NAMES[coco_id]
        try:
            return float(self.net.get_node_value(cat_name)[self.positive_state_idx[cat_name]])
        except:
            return 0.0

    def get_prior(self, coco_id):
        if coco_id not in COCO_NAMES: return 0.0
        return self.prior_cache.get(COCO_NAMES[coco_id], 0.0)

    def clear_evidence(self):
        """
        清除 BN 網路上所有的證據，
        供建構 S_BN 靜態矩陣時的跨類別重置使用。
        """
        # 注意：請確認你的 PySMILE 網路變數名稱。
        # 如果你初始化時是寫 self.net = pysmile.Network()，就用 self.net
        # 如果是 self.network = ... 就改成 self.network.clear_all_evidence()
        self.net.clear_all_evidence()

