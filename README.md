# LEGO Set Price Prediction with ANN

A machine learning project to predict missing US retail prices for LEGO sets using an Artificial Neural Network (scikit-learn), built on an extensive exploratory data analysis of a public LEGO sets dataset.

## Authors

- Luís Augusto Coelho de Souza
- Guilherme Schnekenberg Teixeira

## Project Overview

This project began as a straightforward imputation task, fill in missing `US_retailPrice` values in a dataset of LEGO sets, but EDA revealed the dataset conflates several structurally different product types (buildable sets, books, merchandise, blind-bag bundles, educational kits), each with distinct pricing logic, and that missingness itself is driven by a historical data-collection boundary rather than random chance.

Rather than imputing blindly across this heterogeneous population, the project scopes the model to standard buildable sets released in 2007 or later, backed by evidence gathered throughout the EDA process, and trains an ANN to predict retail price from a set's pieces, minifigs, age range, year, and theme.

## Key Results

| Metric | Validation | Test |
|---|---|---|
| R² | 0.938 | 0.948 |
| MAE | $7.23 | $7.36 |
| RMSE | $14.60 | $15.22 |

The final model explains ~95% of price variance on unseen data and predicts within ~$7 on average.

## Repository Structure

```
├── lego_price_prediction.ipynb   # Full EDA, imputation, and modeling pipeline
└── README.md
```

## Methodology

1. **Missingness analysis**: nullity correlation matrices and pairwise missingness breakdowns to understand **why** data was missing, not just how much.
2. **Data scoping**: the dataset was filtered to `category == 'Normal'` (standard buildable sets) and `year >= 2007`, based on evidence that:
   - Non-`Normal` categories (Gear, Books, Random bundles, etc.) don't have a meaningful piece count, and their pricing follows different logic entirely.
   - Prices for sets before ~2005 are almost entirely unrecorded — a genuine historical data-availability gap rather than a fixable collection issue.
   - The `Education` theme mixes full buildable kits with individually-priced electronic components and was excluded for the same reason.
3. **Feature engineering**: log-transforms for right-skewed features (`pieces`, `minifigs`, `price`), rare-category bucketing for `theme` (categories with fewer than 10 sets grouped into `"Other"`), and one-hot encoding for `theme_grouped`/`themeGroup`.
4. **Imputation**: remaining feature gaps (`pieces`, `minifigs`, `agerange_min`) were filled using scikit-learn's `IterativeImputer` (MICE-style), fit only on training data to avoid leakage.
5. **Modeling**: an `MLPRegressor` (ANN) was trained on `log_price` using a leakage-safe train/validation/test split (70/15/15), with all preprocessing (imputation, scaling, encoding) wrapped in a single `ColumnTransformer` pipeline.
6. **Prediction**: the trained model was applied to the 1,740 rows with genuinely missing prices, with results validated against distributional checks and a manual spot-check against a real-world market listing.

## Data Source

The dataset used in this project is sourced from [Brickset](https://brickset.com), and it was built for Maven Analytics' [Maven LEGO Challenge](https://mavenanalytics.io/data-playground/lego-sets)

## Limitations

- The model is scoped to standard buildable sets (`Normal` category) from 2007 onward; it does not generalize to Gear, Books, or other non-standard product types, or to sets released before 2007.
- Prediction accuracy decreases for high-priced, low-frequency sets, a consequence of training on a right-skewed target distribution.
- `subtheme` was excluded due to high cardinality and missingness; it may carry unexplored residual signal.

## Future Work

- Extend modeling to other product categories with feature sets suited to their actual pricing drivers.
- Explore additional external pricing sources for pre-2005 sets.
- Compare ANN performance against gradient-boosted tree models, particularly for the high-price tail.