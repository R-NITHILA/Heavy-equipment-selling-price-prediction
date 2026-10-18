# Heavy Equipment Selling Price Prediction

Predicting the selling price of heavy industrial machinery with a leakage-safe gradient-boosting ensemble (LightGBM + XGBoost + CatBoost).

Solution for the Kaggle competition: [Heavy Equipment Selling Price Prediction Challenge](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge)

| | |
|---|---|
| **Task** | Tabular regression |
| **Metric** | RMSLE |
| **Best CV score** | **0.2060** (5-fold GroupKFold, weighted ensemble) |
| **Kaggle leaderboard** | **0.1983** |

---

## Problem

Each row is a finalized transaction for a piece of heavy machinery, with system metadata, geographic indicators, component specifications and mechanical configuration. The goal is to predict `TargetValue`, the selling price.

## Dataset

| File | Description |
|---|---|
| `train.csv` | 138,701 rows × 50 columns, including the target |
| `test.csv` | Rows to predict |
| `metadata.csv` | Column descriptions |
| `sample_submission.csv` | Submission format (`TransactionID`, `TargetValue`) |

Key characteristics:

- **Target** ranges from $7,500 to $142,000 and is right-skewed (skew 0.97). A `log1p` transform brings skew to -0.20, so the model is trained on `log1p(TargetValue)` and predictions are inverted with `expm1`. This also aligns training with the RMSLE metric.
- **44 of 50 columns are categorical/object**, many with high cardinality (for example `ProductConfigID` has about 3,600 unique values).
- **Heavy missingness.** Several columns are over 75% null.
- **Repeated machines.** 138,701 transactions map to 120,111 unique `AssetID`s.

## Approach

### 1. EDA and data cleaning

- Converted placeholder strings (`"None or Unspecified"`, `"Unspecified"`) to `NaN`.
- **Structural null analysis.** A null-rate heatmap by `InventoryGroupCategory` showed that most high-null columns are null because the attribute does not apply to that machine type, not because data is missing at random. These columns are kept so the tree models can use "present vs. absent" as signal. Only columns that are almost entirely null across every category were dropped.
- Parsed measurement columns stored as strings with feet/inch/unit markers (`col7`, `col15`, `col21`, `col22`) into numeric inches/values.
- Treated `ManufactureYear == 1001` as a null placeholder (about 14.7k rows).
- Normalized text case (for example `AutoShift` vs `Autoshift`).
- Dropped identifiers and redundant columns (`TransactionID`, `AssetID`, `Spec_FullDescriptor`, `InventoryGroupDescription`) and a few low-importance columns.
- Stripped the redundant prefix from `FunctionalClassification`, keeping only the measurement part.

### 2. Leakage-safe validation

Because the same `AssetID` appears in multiple rows, a random K-fold would let the model see the same machine in both train and validation. All validation uses **`GroupKFold` (5 folds) grouped by `AssetID`**, so scores reflect performance on unseen machines. `AssetID` itself is never used as a feature.

### 3. Preprocessing pipeline

Everything is implemented as scikit-learn transformers, so each fold is fit only on its own training split:

| Step | Purpose |
|---|---|
| `DataCleaner` | Drops columns, fixes `ManufactureYear` placeholder, trims `FunctionalClassification` |
| `ColumnTransformer` + `SimpleImputer` | Mode-imputes `RegionCode` (about 0.4% null) |
| `Encoder` | Groups rare categories (fewer than 50 rows) into `"Other"` for high-cardinality columns |
| `FeatureEngineer` | Creates date and age features (below) |
| `CategoricalTypeFixer` | Fixes the category vocabulary from train and routes unseen values to `"Other"` |

Engineered features: `HasValidManufactureYear`, `TransactionYear`, `TransactionQuarter`, `AssetAge`, `HoursPerYear`.

Remaining nulls are left as-is and categoricals are passed natively to the tree models. No scaling or one-hot encoding is used.

### 4. Models

- **Baseline:** linear regression on median-imputed numeric columns.
- **Gradient boosting:** LightGBM, XGBoost and CatBoost, first with defaults and then tuned with `RandomizedSearchCV` under the same GroupKFold.
- **Ensemble:** weighted average on the log scale, with weights chosen by grid search on out-of-fold predictions.

Tuned hyperparameters:

| Model | Parameters |
|---|---|
| LightGBM | `n_estimators=1200`, `max_depth=6`, `learning_rate=0.1`, `subsample=1.0`, `colsample_bytree=0.9` |
| XGBoost | `n_estimators=1200`, `max_depth=8`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.7` |
| CatBoost | `iterations=1600`, `depth=10`, `learning_rate=0.08`, `l2_leaf_reg=5` |

Final ensemble weights: **LightGBM 0.2, XGBoost 0.4, CatBoost 0.4**.

## Results

Out-of-fold RMSLE (5-fold GroupKFold, lower is better):

| Model | Default | Tuned |
|---|---|---|
| Linear regression (baseline) | 0.6447 | n/a |
| LightGBM | 0.2269 | 0.2122 |
| XGBoost | 0.2204 | 0.2102 |
| CatBoost | 0.2261 | 0.2110 |
| Equal-weight ensemble | n/a | 0.2062 |
| **Weighted ensemble (0.2 / 0.4 / 0.4)** | n/a | **0.2060** |

Observations:

- Gradient boosting cuts error by about 3x relative to the linear baseline, which suggests strongly non-linear relationships (no feature correlates with the target above 0.5).
- Tuning gave a larger gain than switching between models, and the ensemble adds a further small improvement.
- Fold-to-fold standard deviation is around 0.001 to 0.002, so the CV estimate is stable.
- The most important features across the three models include `FunctionalClassification`, `AssetAge`, `ProductConfigID`, `Spec_BaseClass`, `Spec_SubClass` and `CabinType`.

## Repository structure

```
.
├── notebook.ipynb      # Full pipeline: EDA, preprocessing, CV, tuning, ensemble, submission
├── README.md
└── requirements.txt
```

The data is not included in this repository. Download it from the [competition page](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge/data) and place the CSVs where the notebook expects them (it reads from `/kaggle/input/...` by default, so update the paths if running locally).

## Getting started

```bash
git clone https://github.com/R-NITHILA/Heavy-equipment-selling-price-prediction.git
cd Heavy-equipment-selling-price-prediction
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Notes:

- CatBoost is configured with `task_type='GPU'`. If you have no GPU, change it to `'CPU'` (or remove the argument).
- The hyperparameter search cells are commented out, and the best parameters found are hard-coded in the validation and final-fit cells. Uncomment the search cells to re-run tuning.
- Running the final cell writes `submission.csv` in the Kaggle submission format.

## Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · CatBoost · matplotlib · seaborn

## Possible improvements

- Target or frequency encoding for high-cardinality categoricals as an alternative to rare-category grouping.
- Stacking with a meta-learner instead of a fixed weighted average.
- More extensive Bayesian hyperparameter search (for example: Optuna).
- Additional interaction features such as machine type × age or hours.

## License

This project's code is released under the [MIT License](LICENSE). The competition data is not included and remains subject to the Kaggle competition rules
