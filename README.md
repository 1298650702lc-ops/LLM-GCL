# LLM-GCL

LLM-GCL 是一个面向小样本、不平衡二分类任务的多阶段级联学习模型。本仓库提供论文正式模型的独立实现，不包含数据集和实验结果。

## 模型流程

1. 训练 XGBoost、Logistic Regression 和 CatBoost 基础成员。
2. 使用多指标 guardrail 筛选稳定成员。
3. 通过贝叶斯优化确定三类模型的融合权重。
4. 根据 LR 与 XGBoost 的预测分歧进行分区概率校正。
5. 分析错误模式并搜索局部校正规则。
6. 联合优化最终主规则、辅助规则和分类阈值。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 训练

```powershell
python run.py train `
  --data "D:\path\to\train.csv" `
  --external-data "D:\path\to\external.csv" `
  --output-dir "D:\path\to\model_output"
```

训练数据必须包含二元标签列 `左旋标签`。`--external-data` 可省略。

## 预测

```powershell
python run.py predict `
  --model "D:\path\to\model_output\llm_gcl_model.joblib" `
  --data "D:\path\to\new_data.csv" `
  --output "D:\path\to\predictions.csv"
```

完整训练会输出模型文件、模型配置和训练摘要。默认允许不同依赖版本造成的小幅数值差异；如需逐值核验论文参数，可在训练命令中加入 `--strict-signature`。
