# RESULTS AND ANALYSIS 补实验结果报告

## 一、结论

当前最终模型在内部测试集上的AUROC为0.776175，AP为0.332328，召回率为0.923077，特异度为0.500000，F2为0.600000，MCC为0.307140。

论文的RESULTS AND ANALYSIS不需要推倒重建章节结构，但绝大多数表格、图和数值性描述需要替换。Data analysis可作小修改；Robustness statistics和External dataset validation属于中等修改；其余实验性小节均属于大修改。

## 二、逐模块修改判断

| module                                            | change_level   | structure_change   | reason                                                                                             |
|:--------------------------------------------------|:---------------|:-------------------|:---------------------------------------------------------------------------------------------------|
| Baseline comparison                               | 大修改         | 否                 | 全部基线已按4:4:1:1数据职责重新拟合，原表数值、排序和比较句均需替换。                              |
| Data analysis                                     | 小修改         | 否                 | 保留Pearson相关分析和热图结构，仅替换重新计算的相关系数与排序。                                    |
| Interpretability analysis                         | 大修改         | 否                 | SHAP已对当前XGBoost与Logistic Regression融合模型重算，特征排名、方向和审计数据需全部以新结果为准。 |
| Clustering                                        | 大修改         | 局部               | 仍保留六种表示，但两个候选校正未进入最终模型，图题和正文必须标明候选被拒绝。                       |
| Results without minority-class oversampling       | 大修改         | 否                 | 四种协议已在当前训练与调参边界下重跑，旧验证结果不可沿用。                                         |
| Results without guardrail constraints             | 大修改         | 否                 | 全部成员、等权和四种单指标选择均已重新运行，成员构成与结果需要整体替换。                           |
| Stage ablation                                    | 大修改         | 局部               | 分区概率校正和局部概率校正均被门控拒绝，错误模式校正为终端阶段，旧阶段递进叙事不再成立。           |
| Robustness statistics                             | 中等修改       | 否                 | 保留分层Bootstrap方法，结果对象改为内部测试集与外部队列，并替换均值、标准差和置信区间。            |
| External dataset validation                       | 中等修改       | 否                 | 最终模型外部AUROC为0.7105、AP为0.3333，基线结果亦已重跑。                                          |
| Parameter-search strategy and candidate expansion | 大修改         | 否                 | 五种策略已在相同数据边界下重新比较，旧参数搜索表和优劣结论需全部替换。                             |

## 三、各模块新结果表

### 1. Baseline comparison

| method                   | dataset    |   ROC AUC |     AP |   Recall |   Specificity |     F2 |    MCC |
|:-------------------------|:-----------|----------:|-------:|---------:|--------------:|-------:|-------:|
| xgboost                  | validation |    0.7222 | 0.3348 |   0.9231 |        0.5417 | 0.6186 | 0.3351 |
| logistic_regression      | validation |    0.7799 | 0.3907 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| upstream_fusion          | validation |    0.7858 | 0.4136 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| platt_scaling            | validation |    0.7858 | 0.4136 |   0.9231 |        0.5139 | 0.6061 | 0.3163 |
| isotonic_regression      | validation |    0.7399 | 0.285  |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| stacking_lr              | validation |    0.7804 | 0.3929 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| balanced_stacking_lr     | validation |    0.7804 | 0.3936 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| mshse_selective_ensemble | validation |    0.7286 | 0.3277 |   0.9231 |        0.5139 | 0.6061 | 0.3163 |
| final_model              | validation |    0.7473 | 0.3536 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| xgboost                  | test       |    0.6875 | 0.2678 |   0.9231 |        0.5417 | 0.6186 | 0.3351 |
| logistic_regression      | test       |    0.6522 | 0.2065 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| upstream_fusion          | test       |    0.6736 | 0.2245 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| platt_scaling            | test       |    0.6736 | 0.2245 |   0.9231 |        0.5139 | 0.6061 | 0.3163 |
| isotonic_regression      | test       |    0.679  | 0.2297 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| stacking_lr              | test       |    0.6597 | 0.21   |   0.9231 |        0.5    | 0.6    | 0.3071 |
| balanced_stacking_lr     | test       |    0.6629 | 0.2125 |   0.9231 |        0.4861 | 0.5941 | 0.2981 |
| mshse_selective_ensemble | test       |    0.6581 | 0.2107 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| final_model              | test       |    0.7762 | 0.3323 |   0.9231 |        0.5    | 0.6    | 0.3071 |

### 2. Data analysis

| feature      |   pearson_with_outcome |
|:-------------|-----------------------:|
| 性别         |                -0.2765 |
| 身高         |                -0.2126 |
| 甲状腺手术史 |                 0.1758 |
| 体重         |                -0.0912 |
| 糖尿病       |                -0.0575 |
| 结核病史     |                 0.0445 |
| 年龄         |                -0.0374 |
| 左肺上叶     |                 0.0358 |
| 肺部手术史   |                -0.032  |
| 胸部外伤史   |                 0.0309 |
| 右肺上叶     |                 0.0289 |
| 甲状腺疾病   |                -0.0274 |

### 3. Interpretability analysis

|   rank | feature_chinese   | feature_english                   |   mean_abs_shap |   mean_shap |
|-------:|:------------------|:----------------------------------|----------------:|------------:|
|      1 | 性别              | Sex (male = 1)                    |          0.0816 |     -0.0099 |
|      2 | 右肺下叶          | Right lower lobe                  |          0.0087 |     -0.0001 |
|      3 | bmi               | BMI                               |          0.0044 |      0.0015 |
|      4 | 放化疗史          | Radiotherapy/chemotherapy history |          0.004  |      0.004  |
|      5 | 左肺下叶          | Left lower lobe                   |          0.004  |      0.0011 |
|      6 | 右肺上叶          | Right upper lobe                  |          0.0038 |     -0.001  |
|      7 | 右肺中叶          | Right middle lobe                 |          0.003  |      0.0004 |
|      8 | 肺部手术史        | History of pulmonary surgery      |          0.0029 |     -0.0007 |
|      9 | 胸部外伤史        | History of chest trauma           |          0.002  |     -0.0002 |
|     10 | 体重              | Weight                            |          0.002  |      0.0011 |

### 4. Clustering

| representation                         |   clusters |   silhouette |   feature_dimension |   normalized_mean_within_cluster_distance |   normalized_mean_between_cluster_distance |
|:---------------------------------------|-----------:|-------------:|--------------------:|------------------------------------------:|-------------------------------------------:|
| base_learner_probabilities             |          2 |       0.7102 |                  13 |                                    0.3734 |                                     1.792  |
| upstream_fusion                        |          3 |       0.7757 |                   4 |                                    0.2385 |                                     3.224  |
| regional_adjustment_candidate_rejected |          4 |       0.8012 |                   6 |                                    0.1642 |                                     3.6618 |
| clinical_error_structure               |          6 |       0.1997 |                  24 |                                    0.7002 |                                     1.4542 |
| error_pattern_correction_terminal      |          5 |       0.8392 |                   3 |                                    0.1339 |                                     2.0344 |
| local_adjustment_candidate_rejected    |          5 |       0.8134 |                   4 |                                    0.1203 |                                     1.6725 |

### 5. Results without minority-class oversampling

| method                 | dataset    |   ROC AUC |     AP |   Recall |   Specificity |     F2 |    MCC |
|:-----------------------|:-----------|----------:|-------:|---------:|--------------:|-------:|-------:|
| no_oversampling        | validation |    0.7126 | 0.334  |   0.9231 |        0.4583 | 0.5825 | 0.2802 |
| no_oversampling        | test       |    0.7708 | 0.3262 |   0.9231 |        0.4444 | 0.5769 | 0.2714 |
| positive_negative_0_25 | validation |    0.7473 | 0.3536 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| positive_negative_0_25 | test       |    0.7762 | 0.3323 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| positive_negative_0_50 | validation |    0.7361 | 0.3542 |   0.9231 |        0.4722 | 0.5882 | 0.2891 |
| positive_negative_0_50 | test       |    0.7479 | 0.2912 |   0.9231 |        0.4861 | 0.5941 | 0.2981 |
| full_balance_1_00      | validation |    0.7265 | 0.3466 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| full_balance_1_00      | test       |    0.7361 | 0.277  |   0.9231 |        0.5139 | 0.6061 | 0.3163 |

### 6. Results without guardrail constraints

| method               | dataset    |   ROC AUC |     AP |   Recall |   Specificity |     F2 |    MCC |
|:---------------------|:-----------|----------:|-------:|---------:|--------------:|-------:|-------:|
| guardrail            | validation |    0.7473 | 0.3536 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| guardrail            | test       |    0.7762 | 0.3323 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| all_13_family_means  | validation |    0.7286 | 0.3053 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| all_13_family_means  | test       |    0.7083 | 0.2745 |   0.9231 |        0.5139 | 0.6061 | 0.3163 |
| all_13_equal_mean    | validation |    0.6998 | 0.3205 |   0.9231 |        0.5139 | 0.6061 | 0.3163 |
| all_13_equal_mean    | test       |    0.765  | 0.3042 |   0.9231 |        0.5139 | 0.6061 | 0.3163 |
| single_metric_auroc  | validation |    0.7003 | 0.3686 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| single_metric_auroc  | test       |    0.7746 | 0.3919 |   0.9231 |        0.5417 | 0.6186 | 0.3351 |
| single_metric_f2     | validation |    0.7516 | 0.3265 |   0.8462 |        0.5833 | 0.5914 | 0.3094 |
| single_metric_f2     | test       |    0.7318 | 0.3995 |   0.7692 |        0.5556 | 0.5319 | 0.2338 |
| single_metric_recall | validation |    0.7035 | 0.3584 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| single_metric_recall | test       |    0.7756 | 0.3922 |   0.9231 |        0.5417 | 0.6186 | 0.3351 |

### 7. Stage ablation and gating

| method                                 | dataset    |   ROC AUC |     AP |   Recall |   Specificity |     F2 |    MCC |
|:---------------------------------------|:-----------|----------:|-------:|---------:|--------------:|-------:|-------:|
| upstream_fusion                        | validation |    0.7858 | 0.4136 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| upstream_fusion                        | test       |    0.6736 | 0.2245 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| regional_probability_adjustment        | validation |    0.7858 | 0.4136 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| regional_probability_adjustment        | test       |    0.6736 | 0.2245 |   0.9231 |        0.5278 | 0.6122 | 0.3257 |
| error_pattern_correction               | validation |    0.7473 | 0.3536 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| error_pattern_correction               | test       |    0.7762 | 0.3323 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| local_probability_adjustment_candidate | validation |    0.7858 | 0.3976 |   0.7692 |        0.5694 | 0.5376 | 0.244  |
| local_probability_adjustment_candidate | test       |    0.6683 | 0.2194 |   0.6923 |        0.5694 | 0.4891 | 0.1888 |

| stage                                  | accepted   |   objective_gain |   bootstrap_improvement_probability |   seed_support | reasons                                                                                                          |
|:---------------------------------------|:-----------|-----------------:|------------------------------------:|---------------:|:-----------------------------------------------------------------------------------------------------------------|
| regional_probability_adjustment        | False      |           0.0001 |                               0.508 |              3 | objective_gain_below_complexity_margin|bootstrap_support_too_low                                                 |
| error_pattern_correction               | True       |           0.0231 |                               0.834 |              3 | all_pre_registered_gates_passed                                                                                  |
| local_probability_adjustment_candidate | False      |          -0.0195 |                               0.25  |              1 | objective_gain_below_complexity_margin|bootstrap_support_too_low|bo_seed_support_too_low|core_metric_degradation |

### 8. Robustness statistics

| method      | dataset   |   iterations |   point_auc |   bootstrap_mean_auc |   bootstrap_std_auc |   ci95_low |   ci95_high |
|:------------|:----------|-------------:|------------:|---------------------:|--------------------:|-----------:|------------:|
| final_model | test      |         1000 |      0.7762 |               0.7727 |              0.0622 |     0.641  |      0.8836 |
| final_model | external  |         1000 |      0.7105 |               0.7102 |              0.1815 |     0.3158 |      1      |

### 9. External dataset validation

| method                   | dataset   |   ROC AUC |     AP |   Recall |   Specificity |     F2 |     MCC |
|:-------------------------|:----------|----------:|-------:|---------:|--------------:|-------:|--------:|
| xgboost                  | external  |    0.7105 | 0.2436 |      0.5 |        0.7368 | 0.3571 |  0.1539 |
| logistic_regression      | external  |    0.7105 | 0.3333 |      0.5 |        0.4737 | 0.2632 | -0.0155 |
| upstream_fusion          | external  |    0.7105 | 0.3333 |      0.5 |        0.6842 | 0.3333 |  0.1147 |
| platt_scaling            | external  |    0.7105 | 0.3333 |      0.5 |        0.6842 | 0.3333 |  0.1147 |
| isotonic_regression      | external  |    0.5789 | 0.175  |      0.5 |        0.6842 | 0.3333 |  0.1147 |
| stacking_lr              | external  |    0.7105 | 0.3333 |      0.5 |        0.4737 | 0.2632 | -0.0155 |
| balanced_stacking_lr     | external  |    0.7105 | 0.3333 |      0.5 |        0.4737 | 0.2632 | -0.0155 |
| mshse_selective_ensemble | external  |    0.5    | 0.3    |      0.5 |        0.5789 | 0.2941 |  0.0468 |
| final_model              | external  |    0.7105 | 0.3333 |      0.5 |        0.6842 | 0.3333 |  0.1147 |

### 10. Parameter-search strategy and candidate expansion

| method                       | dataset    |   ROC AUC |     AP |   Recall |   Specificity |     F2 |    MCC |
|:-----------------------------|:-----------|----------:|-------:|---------:|--------------:|-------:|-------:|
| semantic_candidate_fixed     | validation |    0.7505 | 0.3648 |   0.9231 |        0.4861 | 0.5941 | 0.2981 |
| semantic_candidate_fixed     | test       |    0.7313 | 0.2706 |   0.9231 |        0.4861 | 0.5941 | 0.2981 |
| semantic_candidate_fixed     | external   |    0.7105 | 0.3333 |   0.5    |        0.5263 | 0.2778 | 0.0155 |
| gated_final_model            | validation |    0.7473 | 0.3536 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| gated_final_model            | test       |    0.7762 | 0.3323 |   0.9231 |        0.5    | 0.6    | 0.3071 |
| gated_final_model            | external   |    0.7105 | 0.3333 |   0.5    |        0.6842 | 0.3333 | 0.1147 |
| bayesian_expanded_candidates | validation |    0.7537 | 0.3503 |   0.8462 |        0.5139 | 0.5612 | 0.2601 |
| bayesian_expanded_candidates | test       |    0.6544 | 0.206  |   0.8462 |        0.5417 | 0.5729 | 0.2794 |
| bayesian_expanded_candidates | external   |    0.5263 | 0.3026 |   0.5    |        0.7368 | 0.3571 | 0.1539 |
| genetic_evidence_candidates  | validation |    0.7879 | 0.353  |   0.9231 |        0.4722 | 0.5882 | 0.2891 |
| genetic_evidence_candidates  | test       |    0.726  | 0.3196 |   0.9231 |        0.4722 | 0.5882 | 0.2891 |
| genetic_evidence_candidates  | external   |    0.6316 | 0.2381 |   0.5    |        0.5263 | 0.2778 | 0.0155 |
| genetic_expanded_candidates  | validation |    0.7431 | 0.3188 |   0.7692 |        0.5139 | 0.5155 | 0.2042 |
| genetic_expanded_candidates  | test       |    0.6554 | 0.2069 |   0.8462 |        0.5417 | 0.5729 | 0.2794 |
| genetic_expanded_candidates  | external   |    0.5526 | 0.5526 |   0.5    |        0.7368 | 0.3571 | 0.1539 |

## 四、实验边界

- train仅用于基础学习器拟合。
- tuning用于补实验中的校准器、融合权重、规则参数、阈值和搜索策略选择。
- validation仅用于Guardrail及单指标成员选择。
- test和external仅用于配置确定后的评估，不参与任何参数或阈值选择。
- 本套件未修改原论文main.tex，也未覆盖原有result_4411冻结产物。
