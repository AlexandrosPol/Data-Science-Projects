# Data Science Projects

## Description
This repository showcases my work on various data science projects, utilizing R and Python for statistical analysis and machine learning. Each project tackles a unique problem, from predictive modeling to pattern recognition in complex datasets.

## Table of Contents
- [Installation](#installation)
- [Running the Scripts](#running-the-scripts)
- [Projects](#projects)
  - [Soil Data Clustering](#soil-data-clustering)
  - [Market Basket Analysis](#market-basket-analysis)
  - [Abalone Age Prediction](#abalone-age-prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Installation
For R projects:
Before running the scripts, ensure that R and RStudio are installed on your system. The scripts rely on several R packages, which can be installed using the following command in the R console:
`R`

install.packages(c("caret", "ggplot2", "cluster", "arules", "AppliedPredictiveModeling", "Metrics"))

For Python Project:

packages used: numpy, pandas, matplotlib, seaborn, scikit-learn

## Running the Scripts
Each script can be executed within RStudio. Set the working directory to the location of the data files or update the file paths in the scripts accordingly.

## Projects
### Soil Data Clustering
This project applies k-means clustering to soil data from Northern Greece to identify patterns and inform better agricultural practices. [View the script]([Machine%20Learning%20Algorithms/Soil%20Dataset%20-%20k%20means%20clustering.R](https://github.com/AlexandrosPol/Data-Science-Projects/blob/main/Abalone%20Age%20Prediction/Abalone%20Dataset%20-%20linear%20regression%2Bpca.R)).


### Market Basket Analysis
This project applies the apriori algorithm to discover patterns and trends in country visit data, revealing insights into traveler behaviors and country associations. [View the script](Machine%20Learning%20Algorithms/Countries%20Dataset%20-%20apriori%20algorithm.R).

### Abalone Age Prediction
Using regression analysis and principal component analysis (PCA), this project aims to predict the age of abalones from their physical measurements, highlighting the impact of dimensionality reduction on predictive accuracy.  [View the script](Machine%20Learning%20Algorithms/Abalone%20Dataset%20-%20linear%20regression%2Bpca.R).

### Titanic Survival Prediction
Analyzing the Titanic dataset to predict survival rates using machine learning in Python. The project demonstrates data cleaning, exploratory data analysis, and the application of classification algorithms.
The objective of the model is to predict whether passengers on the Titanic would have survived, based on features like passenger class, sex, age, number of siblings/spouses, parents/children on board, fare, and port of embarkation. This model could help us understand the factors that contributed to the likelihood of survival during the Titanic disaster.

## Results
Soil Data Clustering: Identified three distinct soil types, providing a basis for customized agricultural approaches.

Market Basket Analysis: Revealed strong association rules that provide insights into the interconnectedness of country visits.

Abalone Age Prediction: The linear regression model, prior to PCA, had a mean squared error (MSE) of 6.297. Post-PCA, with only two principal components used, the MSE slightly increased, indicating a trade-off between model simplicity and accuracy.

Detailed analysis and discussions are available within each script's comments and results sections.

## Contributing
I welcome contributions to this repository. Please fork the project, make your changes, and submit a pull request for review. Collaboration is key to advancing data science!

## License
This repository and its contents are provided under the MIT License. This permissive license allows for sharing and adaptation with appropriate credit given. See the LICENSE.md for full terms.

## Contact
Feel free to reach out to me for questions, discussions, or collaboration proposals via [email](mailto:apolyzoidis@hotmail.com).

## Acknowledgments
Special thanks to the data providers.
