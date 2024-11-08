# Digit Recognizer Project

This project aims to develop a digit recognition model using PyTorch. The goal is to classify handwritten digits (0-9) from the popular MNIST dataset, achieving high accuracy using a Long Short-Term Memory (LSTM) network.

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Tools and Libraries](#tools-and-libraries)
- [Project Steps](#project-steps)
- [Model Details](#model-details)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Objective
The objective of this project is to classify images of handwritten digits using a deep learning model. The dataset consists of grayscale images of handwritten digits with a size of 28x28 pixels.

## Dataset
- Dataset: MNIST handwritten digit dataset.
- Each image is 28x28 pixels, and the labels represent digits from 0 to 9.

## Tools and Libraries
- Programming Language: Python
- Libraries: PyTorch, torchvision, pandas, numpy, matplotlib, seaborn, sklearn, tqdm

## Project Steps
1. **Data Loading and Preprocessing**
   - Loaded the training data from a CSV file into a pandas DataFrame.
   - Converted pixel values to tensors and normalized them.
   - Split the dataset into training and testing subsets.
2. **Model Architecture**
   - Created an LSTM model with the following layers:
     - LSTM Layer with 100 hidden units.
     - Fully connected layer for classification.
   - Optimized using Cross Entropy Loss and SGD.
3. **Model Training**
   - Trained the model over multiple epochs while monitoring accuracy and loss.
   - Visualized the results to understand the model's performance.
4. **Model Evaluation**
   - Evaluated the model using the test dataset.
   - Achieved an accuracy score of 97.83%.

## Model Details
- **Input Size**: 28 (flattened pixel rows)
- **Hidden Dimension**: 100
- **Output Dimension**: 10 (digits 0-9)
- **Optimizer**: SGD with a learning rate of 0.1
- **Loss Function**: Cross Entropy Loss

## How to Run
1. **Clone the Repository**:
   ```
   git clone https://github.com/YourUsername/Digit-Recognizer
   ```
2. **Install Required Packages**:
   ```
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**:
   Open `digit_recognizer.ipynb` in Jupyter Notebook or JupyterLab and execute the cells to train and evaluate the model.

## Results
- **Training Accuracy**: Achieved high accuracy on the training set.
- **Test Accuracy**: Achieved 97.83% accuracy on the test set.
- **Visualization**: Displayed predictions and the corresponding images for validation.

## Future Improvements
- **Data Augmentation**: Implement data augmentation techniques to improve model generalization.
- **Hyperparameter Tuning**: Optimize the LSTM parameters and learning rate for better performance.

## Contact
Feel free to reach out via [GitHub](https://github.com/YourUsername) if you have questions or suggestions.
