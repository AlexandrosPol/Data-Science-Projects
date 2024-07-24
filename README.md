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
  - [Titanic Survival Prediction](#titanic-survival-prediction)
  - [CardsDeck Image Classification](#cardsdeck-image-classification)
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

For Python Projects:

packages used: numpy, pandas, matplotlib, seaborn, scikit-learn, torch, torchvision, timm, tqdm

## Running the Scripts
Each script can be executed either within RStudio (for R projects) and using Jupyter Notebook or another IDE (for python projects).

## Projects
### Soil Data Clustering
This project applies k-means clustering to soil data from Northern Greece to identify patterns and inform better agricultural practices. [View the script](https://github.com/AlexandrosPol/Data-Science-Projects/blob/main/Soil%20Types%20Identification%20Analysis/Soil%20Dataset%20-%20k%20means%20clustering.R).

### Market Basket Analysis
This project applies the apriori algorithm to discover patterns and trends in country visit data, revealing insights into traveler behaviors and country associations. [View the script](https://github.com/AlexandrosPol/Data-Science-Projects/blob/main/Countries%20Visitation%20Analysis/Countries%20Dataset%20-%20apriori%20algorithm.R).

### Abalone Age Prediction
Using regression analysis and principal component analysis (PCA), this project aims to predict the age of abalones from their physical measurements, highlighting the impact of dimensionality reduction on predictive accuracy.  [View the script](https://github.com/AlexandrosPol/Data-Science-Projects/blob/main/Abalone%20Age%20Prediction/Abalone%20Dataset%20-%20linear%20regression%2Bpca.R).

### Titanic Survival Prediction
Analyzing the Titanic dataset to predict survival rates using machine learning in Python. The project demonstrates data cleaning, exploratory data analysis, and the application of classification algorithms.
The objective of the model is to predict whether passengers on the Titanic would have survived, based on features like passenger class, sex, age, number of siblings/spouses, parents/children on board, fare, and port of embarkation. This model could help us understand the factors that contributed to the likelihood of survival during the Titanic disaster. [View the script](https://github.com/AlexandrosPol/Data-Science-Projects/blob/main/Titanic%20Survival%20Prediction/Titanic%20Survival%20Prediction.ipynb).

### CardsDeck Image Classification
This project involves classifying images of playing cards into their respective categories using deep learning. The dataset consists of images of playing cards, and the task is to correctly identify the rank and suit of each card.
The project employs a convolutional neural network (CNN) implemented in PyTorch. The model is trained using a transfer learning approach with a pre-trained network from the timm library, specifically designed for image classification tasks.
Key steps in the project include:

Data Preprocessing: Images are resized, normalized, and augmented to improve the model's generalization capability.

Model Training: A pre-trained model is fine-tuned on the cards dataset. Various hyperparameters such as learning rate, batch size, and number of epochs are optimized.

Evaluation: The model's performance is evaluated using accuracy metric.

The project demonstrates the use of advanced deep learning techniques and transfer learning to tackle an image classification problem. [View the script](https://github.com/AlexandrosPol/Data-Science-Projects/blob/main/Card%20Deck%20-%20Image%20Classification%20with%20PyTorch/card-deck-image-classification-with-pytorch.ipynb)


## Results
Soil Data Clustering: Identified three distinct soil types, providing a basis for customized agricultural approaches.

Market Basket Analysis: Revealed strong association rules that provide insights into the interconnectedness of country visits.

Abalone Age Prediction: The linear regression model, prior to PCA, had a mean squared error (MSE) of 6.297. Post-PCA, with only two principal components used, the MSE slightly increased, indicating a trade-off between model simplicity and accuracy.

Titanic Survival Prediction: The best-performing model achieved an accuracy of 81%, highlighting key factors affecting survival chances.

Detailed analysis and discussions are available within each script's comments and results sections.

CardsDeck Image Classification: Achieved an accuracy of 97% on the validation set, effectively classifying the rank and suit of playing cards.

## Contributing
I welcome contributions to this repository. Please fork the project, make your changes, and submit a pull request for review. Collaboration is key to advancing data science!

## License
This repository and its contents are provided under the MIT License. This permissive license allows for sharing and adaptation with appropriate credit given. See the LICENSE.md for full terms.

## Contact
Feel free to reach out to me for questions, discussions, or collaboration proposals via [email](mailto:apolyzoidis@hotmail.com).

## Acknowledgments
Special thanks to the data providers.
