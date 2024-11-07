# Card Deck Image Classification

## Objective
Classify images of playing cards into their respective categories (suit and rank) using a Convolutional Neural Network (CNN) implemented in PyTorch. The model is trained to accurately identify each card's rank and suit.

## Dataset
- **Structure**: The dataset contains images of playing cards organized in folders by class.
- **Classes**: 53 unique classes, covering all combinations in a standard card deck.
- **Dataset Organization**:
  - Training, validation, and test sets are separated to evaluate model performance.

## Tools and Libraries
- **Programming Language**: Python
- **Libraries**:
  - `torch` and `torchvision` for deep learning and data processing.
  - `matplotlib` and `PIL` for visualization.
  - `tqdm` for tracking model training progress.

## Model Architecture
- **Base Model**: EfficientNet-B0 pretrained on ImageNet, modified for 53 classes.
- **Layers**:
  - The EfficientNet layers are used as a feature extractor.
  - A custom fully connected layer is added for classification into 53 unique card classes.
- **Training Strategy**: Fine-tuning only the final layers while keeping other layers frozen to leverage pre-trained weights.

## Project Workflow

### 1. Data Preparation
   - **Image Transformations**: Resized to 128x128 pixels and normalized.
   - **Custom Dataset Class**: Implemented a `PlayingCardDataset` class to handle image loading and preprocessing for efficient data handling.

### 2. Model Training
   - **Training and Validation**: Split data into training and validation sets.
   - **Loss Function**: CrossEntropyLoss for multi-class classification.
   - **Optimizer**: Adam optimizer with a learning rate of 0.001.
   - **Epochs**: Model trained over 10 epochs to achieve convergence.

### 3. Evaluation and Results
   - **Accuracy**: Achieved a test accuracy of 96.98%.
   - **Loss Tracking**: Monitored training and validation loss to ensure convergence.
   - **Visualization**: Plotted training and validation loss over epochs for performance tracking.

### 4. Inference and Prediction Visualization
   - **Prediction Function**: Loads and preprocesses new images for model prediction.
   - **Visualization**: Displays the original image, predicted class, and confidence score.

## Results
- **Test Accuracy**: 96.98%
- **Loss Plot**:
  ![Loss over Epochs](images/loss_over_epochs.png)
- **Sample Predictions**:
  - ![Prediction 1](images/sample_prediction1.png)
  - ![Prediction 2](images/sample_prediction2.png)

## How to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Card-Deck-Classification
   ```
2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Training**:
   - Open `Card_Deck_Classification.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells in the notebook to train the model.

4. **Inference**:
   - Use the inference section in the notebook to test predictions on new images.

## Future Work
- **Data Augmentation**: Apply augmentations to improve the model's ability to generalize to new data.
- **Hyperparameter Tuning**: Experiment with different model architectures and optimizers to enhance performance.
- **Evaluation Metrics**: Add additional metrics like precision, recall, and F1-score for a more comprehensive evaluation of the model's performance.
