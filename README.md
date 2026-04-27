# Kaggle Playground S6E4 — Predicting Irrigation Need

A multi-class classification problem (High/Low/Medium irrigation need) scored on balanced accuracy. The dataset is generated from a deep learning model trained on a small original dataset of ~10k rows.

Final score: **0.98023** (inside the public-LB plateau, up from 0.96001 at v1).

## Key findings

The target was generated from a deterministic rule on 6 features (4 thresholds + 2 stage/mulch dummies), with ~1.85% label noise added by the generator. Initially, the other 6 categorical columns look like noise, but they truned out to be important.
While the noisy categoricals seemed useless at first, they contaiend some signal that helped break the 0.97 treshold.
The generator also distorted feature distributions, especially `Rainfall_mm` (KS = 0.158 vs original). An "original-only" trained model performs worse on synthetic test (0.957) than a model trained directly on synthetic data (0.967), a distribution shift.

## Two phases of the journey

**Phase 1 (v1 → v11): incremental gains on a small feature set, 0.96001 → 0.97061.** Mostly conservative methodology — class weights, K-fold, stacking, bounded tuning. Each step verified before moving on. Hit a ceiling at 0.97 because I over-pruned features.

**Phase 2 (v12): big jump from inspecting a public 0.98109 OOF notebook, 0.97061 → 0.98023.** A community member published a notebook hitting OOF 0.98109 with a single XGBoost. Apparantly a much simpler approche did much better than me.
What they did differently was mass feature engineering plus *in-fold multiclass target encoding on every column*, including the categoricals I had dropped as noise. 
The public notebook by [rawashishsin](https://www.kaggle.com/code/rawashishsin/s6e4-highest-score-xgboost-cv-0-98109) gave me the feature-engineering and target-encoding pipeline that my final version used, I do not think I would have broken 0.98 without it.

## What worked

- **In-fold multiclass target encoding** of every column (~150 engineered features × 3 classes ≈ 450 TE features). Very important for 0.98
- **Mass feature engineering**
- **Class-balanced sample weights** for balanced accuracy.
- **Shallow trees**. Optuna independently picked depth 5 across XGBoost, LightGBM, and CatBoost in phase 1 (vs my hand-tuned depth 7-8). Went shallower still (depth 3), inspired by other notebooks.
- **K-fold + seed averaging**. Standard variance reduction. Per-fold std stayed tight (~0.001) throughout. +0.001 LB.
- **Logistic regression stacking**. A simple LR meta-learner over XGB/LGB/CB OOF probabilities beat equal-weight averaging at every altitude tested (0.967, 0.970, 0.978). +0.001 LB consistently.
- **Engineered meta-features** in the stacking input (per-class std/var, agreement, entropy, margin). +0.0002 LB.
- **Bounded post-hoc class-weight tuning** on softmax outputs, in [0.7, 1.5]. Worked when models were poorly calibrated.
- **Pseudo-labeling** (top 50% confident test rows, soft weight 0.5) for one extra K-fold retrain. Small win(+0.0004 LB). Did not use in the end.

## What didn't work

- **Adding diverse base models** (logistic regression, MLP, RealMLP). All three solo-scored 0.962-0.965.
- **Recalibrating the rule on synthetic data**. 
- **Hill climbing meta-learner**
- **Knowledge distillation** 
- **Aggressive class-weight tuning**

## Final pipeline (v12)

```
features:    full feature pool (~150 engineered):
             - thresholds, domain ratios, formula score
             - bigrams + trigrams of top categoricals
             - binned-numerical × categorical interactions
             - per-categorical mean/diff/ratio of top numerics
             - round/digit/decimal of top numerics
             - pairwise factorized features
encoding:    in-fold multiclass target encoding on every feature (3 cols per feature),
             with sklearn TargetEncoder, smooth='auto', cv=5
models:      XGBoost (depth 3, max_bin 1100, n_estimators 2600),
             LightGBM (num_leaves 15), CatBoost (depth 4)
training:    5-fold stratified CV × 3 seed averaging, class-balanced sample weights,
             original dataset concatenated with weight 0.5
stacking:    logistic regression meta-learner on K-folded OOF probabilities,
             plus engineered meta-features (per-class std/var, agreement, entropy, margin)
post-hoc:    bounded multi-seed CV class-weight tuning, in [0.7, 1.5]
```

## Running the notebook

Run `waterworld.ipynb`. 

Data: the competition train/test files plus the [original irrigation dataset](https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset).

## Score progression

| version | LB | what changed |
|---|---|---|
| v1 | 0.96001 | vanilla XGBoost baseline |
| v3 | 0.96587 | class-balanced sample weights for BA |
| v4 | 0.96693 | extreme (overfit) class-weight tuning |
| v8 | 0.96927 | Optuna hyperparameters + K-fold + seed averaging + bounded tuning |
| v9 | 0.97023 | LR meta-learner stacking |
| v10 | 0.96957 | tried adding LR + MLP + RealMLP — regressed |
| v11 | 0.97061 | pseudo-labeling + meta-features in stack |
| v12 | **0.98023** | mass FE + in-fold target encoding (inspired by [rawashishsin](https://www.kaggle.com/code/rawashishsin/s6e4-highest-score-xgboost-cv-0-98109)) + 3-model stack + tuning |
