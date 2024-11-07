# Titanic Survival Prediction Project

## Objective
- Predict whether passengers on the Titanic would have survived, based on features like class, age, sex, family aboard, and fare.
- Provide insights into social and historical patterns during the Titanic disaster.

## Dataset
- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)
- **Features**: Passenger class, age, sex, family aboard, ticket fare, port of embarkation.

## Tools
- **Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn

## Process
1. ### Define the Problem
   - Set goals for predictive modeling based on Titanic survival data.

2. ### Data Collection and Understanding
   - Loaded the dataset into a pandas DataFrame and explored each feature.

3. ### Data Wrangling and Preprocessing
   - Dropped irrelevant columns (`PassengerId`, `Name`, `Ticket`).
   - Handled missing values in `Age` and `Embarked` columns.
   - Converted categorical variables (`Sex`, `Embarked`) to numeric.

4. ### Exploratory Data Analysis (EDA)
   - Visualized survival distributions across features (e.g., sex, class, family size).
   - Gained insights on factors influencing survival.

5. ### Feature Engineering
   - Binned ages into categorical groups.
   - Encoded categorical data.
   - Analyzed correlations with a heatmap.

6. ### Model Selection and Training
   - Compared algorithms: Logistic Regression, Decision Tree, SVM, and K-Nearest Neighbor.
   - Used GridSearch for Decision Tree hyperparameter tuning.

7. ### Model Evaluation
   - **Best Model**: Decision Tree with highest validation accuracy (81%).
   - Evaluated overfitting risk and tested model on a separate test set.

## Results and Insights
- **Key Factors for Survival**:
  - Higher survival rates for females, younger passengers, and higher-class ticket holders.
  - Small family sizes were slightly advantageous for survival.
- **Model Insights**:
  - The Decision Tree was selected for interpretability and a balance between accuracy and complexity.

## Conclusion
- Historical insights into socio-economic factors and biases affecting survival.
- Implications for safety protocols and potential future research in social demographics during disasters.

---

This project demonstrates the application of machine learning to a historically significant dataset, providing valuable insights and a case study in predictive modeling.
