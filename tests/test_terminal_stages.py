import copy
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_gcl import training
from llm_gcl.model import LLMGCLModel
from llm_gcl.rules import group_probability, segmented_probability, three_group_probability


class TerminalStagesTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"weight": [60., 50., 70., 40.], "aux": [1, 0, 0, 1]})
        self.xgb = np.array([.2, .3, .6, .7])
        self.lr = np.array([.6, .7, .2, .3])
        self.upstream = (self.xgb + self.lr) / 2
        self.rule = {"feature": "weight", "rule_type": "numeric_gt", "split_value": 58.}
        self.config = {
            "selected_member_ids": {"xgb": [], "lr": [], "cat": []},
            "upstream": {"weights": {"xgb": .5, "lr": .5, "cat": 0.}, "threshold": .17},
            "round8": {"method": "bayesian_optimization", "delta": .1, "t_lr": .2, "t_mid": .3, "t_xgb": .4, "threshold": .4},
            "round9": {"rule": self.rule, "threshold_group": .2, "threshold_other": .3, "threshold": .6},
            "round10_candidate": {"primary_rule": self.rule, "aux_rule": {"feature": "aux", "rule_type": "numeric_gt", "split_value": .5}, "threshold_primary_aux": .15, "threshold_primary_base": .25, "threshold_other": .35, "threshold": .7},
        }

    def family(self, model, frame, family):
        return {"xgb": self.xgb, "lr": self.lr, "cat": np.zeros(4)}[family]

    def test_prediction_and_serialization_for_all_terminals(self):
        for regional in (False, True):
            for terminal in ("upstream", "round8", "round9", "round10"):
                if terminal == "round8" and not regional:
                    continue
                with self.subTest(regional=regional, terminal=terminal):
                    config = copy.deepcopy(self.config)
                    config["evaluation_terminal_stage"] = terminal
                    if not regional:
                        config["round8"] = {"method": "identity_no_round8", "threshold": .17}
                    r8 = segmented_probability(self.upstream, self.lr-self.xgb, .1, .2, .3, .4)[0] if regional else self.upstream
                    expected = {
                        "upstream": self.upstream,
                        "round8": r8,
                        "round9": group_probability(r8, np.array([True, False, True, False]), .2, .3),
                        "round10": three_group_probability(r8, np.array([True, False, True, False]), np.array([True, False, False, True]), .15, .25, .35),
                    }[terminal]
                    threshold = config["round10_candidate" if terminal == "round10" else terminal]["threshold"]
                    # Rejected later-stage configurations must never be required for inference.
                    if terminal in ("upstream", "round8", "round10"):
                        del config["round9"]
                    model = LLMGCLModel(list(self.frame), {}, config)
                    with patch.object(LLMGCLModel, "_family_probability", lambda m, f, fam: self.family(m, f, fam)):
                        np.testing.assert_allclose(model.predict_proba(self.frame), expected)
                        self.assertEqual(model.threshold, threshold)
                        np.testing.assert_array_equal(model.predict(self.frame), expected >= threshold)
                        with tempfile.TemporaryDirectory() as tmp:
                            model.save(Path(tmp)/"model.joblib")
                            loaded = LLMGCLModel.load(Path(tmp)/"model.joblib")
                            np.testing.assert_allclose(loaded.predict_proba(self.frame), expected)

    def test_training_saves_the_actual_terminal_and_metrics(self):
        split = SimpleNamespace(feature_columns=list(self.frame))
        for role in ("train", "tuning", "validation", "test"):
            setattr(split, role+"_meta", self.frame)
            setattr(split, "y_"+role, pd.Series([0, 1, 0, 1]))
        selection = {"selected_xgb_member_ids": [], "selected_lr_member_ids": [], "selected_cat_member_ids": []}
        for terminal in ("upstream", "round8", "round9", "round10"):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                stages = {name: {"config": {**copy.deepcopy(self.config[key]), "tuning_metrics": {"sentinel": name}}} for name, key in [("upstream", "upstream"), ("round8", "round8"), ("round9", "round9"), ("round10", "round10_candidate")]}
                replacements = {
                    "load_labeled_dataframe": self.frame, "build_split_4411": split,
                    "train_base_pool": (pd.DataFrame(), {}), "select_xgb_lr": selection, "select_cat": selection,
                    "raw_upstream_frames": {}, "_attached_frames": {"tuning": self.frame},
                    "candidate_rules": [self.rule], "identity_round8": stages["upstream"],
                }
                replacements.update({"run_"+name: value for name,value in stages.items()})
                for name, value in replacements.items():
                    stack.enter_context(patch.object(training, name, return_value=value))
                stack.enter_context(patch.object(training, "_evaluate_stage_gate", side_effect=lambda name,*a,**k: {"stage": name, "accepted": name == terminal}))
                stack.enter_context(patch.object(LLMGCLModel, "evaluate", return_value={}))
                model, summary = training.train_formal_model("unused.csv", tmp)
                self.assertEqual(model.config["evaluation_terminal_stage"], terminal)
                self.assertEqual(summary["tuning_metrics"], {"sentinel": terminal})
                self.assertTrue((Path(tmp)/"llm_gcl_model.joblib").exists())

    def test_strict_signature_still_rejects_other_terminals(self):
        config = copy.deepcopy(self.config)
        config["selected_member_ids"] = {"xgb": ["xgb_member_04"], "lr": ["lr_member_01"], "cat": []}
        for terminal in ("upstream", "round8", "round10"):
            config["evaluation_terminal_stage"] = terminal
            with self.assertRaisesRegex(RuntimeError, "终止阶段"):
                training.verify_4411_signature(config)


if __name__ == "__main__":
    unittest.main()
