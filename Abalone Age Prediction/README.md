# Abalone Age Prediction Using Linear Regression and PCA

## Objective
To predict the age of abalone (represented by the 'Rings' variable) using linear regression based on physical measurements and to demonstrate feature selection and dimensionality reduction techniques.

## Tools
- **Language**: R
- **Libraries**: AppliedPredictiveModeling, caret, Metrics

## Dataset Overview
- **Data Source**: The abalone dataset from the AppliedPredictiveModeling package.
- **Description**: The dataset includes physical measurements of abalone, with the target variable being the number of rings, which indicates the age.

## Process

1. **Data Preparation**: 
   - Loaded the abalone dataset.
   - Excluded non-predictive variables like 'Type' (categorical) and 'Rings' (target) from the predictor variables.

2. **Feature Importance Analysis**:
   - Ranked variables by importance using a filter-based approach to assess their contribution to predicting age.

3. **Linear Regression Modeling**:
   - Built a linear regression model using the top-ranked predictors.
   - Evaluated the model using Mean Squared Error (MSE) as a performance metric.

4. **Dimensionality Reduction (PCA)**:
   - Scaled data for PCA and computed principal components.
   - Built a simplified linear regression model using only the first two principal components.
   - Compared model performance with and without PCA to assess interpretability vs. accuracy.

## Modeling & Evaluation

- **Initial Model** (using top predictors):
  - **MSE**: 6.297
  - **Interpretation**: Baseline model using top-ranked variables provides good predictive accuracy.

- **PCA Model** (using first two principal components):
  - **MSE**: 6.645
  - **Interpretation**: PCA-based model enhances interpretability with fewer variables but at a slight cost to predictive accuracy.

## Key Findings
- **Feature Selection**: Top-ranked variables provided a strong predictive basis for the initial model.
- **PCA Impact**: PCA simplified the model but slightly increased MSE, indicating a trade-off between interpretability and predictive performance.

## Conclusion
This analysis illustrates the application of linear regression for age prediction, enhanced by feature selection and PCA for dimensionality reduction. While PCA can reduce model complexity, it may affect prediction accuracy.
