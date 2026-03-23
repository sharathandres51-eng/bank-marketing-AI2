# Bank Marketing — Term Deposit Subscription Prediction

Final project for Artificial Intelligence II. The goal is to predict whether a customer will subscribe to a term deposit based on data from a Portuguese bank's direct marketing campaigns.

---

## Dataset

Using the full version of the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`) — 41,188 records and 20 features. This version includes five macroeconomic indicators (Euribor 3-month rate, consumer confidence index, employment variation rate, etc.) that aren't in the original smaller dataset.

Class split is roughly 89% No / 11% Yes, which makes this a fairly challenging imbalance problem.

One important note: the `duration` column (call length) was dropped entirely. It's heavily predictive but only known after the call ends, so keeping it would make the model useless in any real deployment scenario.

---

## What's in the notebook

**Data exploration & cleaning**
- Class distribution, subscription rates by job, age, month, previous outcome
- Macro feature correlation analysis — emp.var.rate, euribor3m, and nr.employed turned out to be highly collinear (r > 0.90), so only euribor3m and cons.conf.idx were kept

**Feature engineering**
- Log-transformed campaign contacts (right-skewed)
- Contact recency buckets from pdays
- Age groups and age-job interaction segments
- High-value customer flag
- Relationship quality score
- Interaction terms between economic indicators and customer wealth proxies

**Models trained**
- Logistic Regression (baseline + tuned)
- Random Forest
- XGBoost
- CatBoost (handles categorical features natively)

All models tuned with GridSearchCV, 5-fold stratified CV, optimising for minority class F1.

**Imbalance handling**
- `class_weight='balanced'` for LR and RF
- `scale_pos_weight` for XGBoost and CatBoost
- SMOTE oversampling applied to training set

**Evaluation**
- Minority class F1, ROC-AUC, PR-AUC
- Confusion matrices, ROC curves, PR curves
- Decision threshold sweep across all models
- Cost-sensitive threshold analysis: with call cost €5 and subscription margin €100, breakeven precision is just 5% — optimal threshold ends up around 0.33 rather than the default 0.5

---

## Results (test set)

| Model | Minority F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.449 | 0.799 | 0.446 |
| Random Forest | ~0.47 | ~0.80 | ~0.46 |
| XGBoost | 0.494 | 0.814 | 0.487 |
| CatBoost | ~0.53 | ~0.82 | ~0.49 |

XGBoost at threshold 0.33 gives an estimated net campaign value of ~€57,755 on the test set.

---

## Setup

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn xgboost catboost imbalanced-learn matplotlib seaborn
```

Then just run `bank_marketing_project.ipynb` top to bottom. CatBoost tuning takes a while (~20–30 min depending on your machine).

---

## Files

```
bank_marketing_project.ipynb   # main notebook
bank-additional-full.csv       # dataset
bank-additional-names.txt      # feature descriptions
```

---

## Limitations worth knowing

- CV scores are slightly optimistic because hyperparameter tuning and cross-validation both use the same training data (nested CV would fix this but is overkill here)
- SMOTE is applied before CV splits, so validation folds see synthetic samples — test set results are the more reliable performance estimate
- Macro features like Euribor have very high correlation with each other; dropping all but two is a judgment call, PCA would be the cleaner solution
