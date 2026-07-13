from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import read_csv
from .model import LLMGCLModel
from .training import train_formal_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="论文正式 LLM-GCL 模型")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="从基础模型开始执行完整训练与贝叶斯优化链")
    train.add_argument("--data", required=True, help="含左旋标签的内部 CSV")
    train.add_argument("--output-dir", required=True, help="模型输出目录；请放在本源码目录之外")
    train.add_argument("--external-data", help="可选：含左旋标签的外部验证 CSV")
    train.add_argument("--strict-signature", action="store_true", help="要求成员、规则和所有连续参数逐值复现论文，否则中止")

    predict = subparsers.add_parser("predict", help="使用训练好的模型预测")
    predict.add_argument("--model", required=True, help="llm_gcl_model.joblib 路径")
    predict.add_argument("--data", required=True, help="待预测 CSV")
    predict.add_argument("--output", required=True, help="预测 CSV 输出路径")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        train_formal_model(args.data, args.output_dir, args.external_data, strict_signature=args.strict_signature)
        return
    model = LLMGCLModel.load(args.model)
    frame = read_csv(args.data)
    probability = model.predict_proba(frame)
    output = pd.DataFrame({"y_prob": probability, "y_pred": model.predict(frame), "threshold": float(model.config["round10"]["threshold"])})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, encoding="utf-8-sig")
    print(f"[完成] 预测已保存到：{destination.resolve()}")
