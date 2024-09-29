#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ====================
# Credit Card Fraud Detection using Deep Learning
# Author: Alexandros Polyzoidis
# ====================
# This project uses a deep learning neural network in PyTorch to detect fraudulent 
# credit card transactions. We aim to build a robust binary classifier that 
# handles the class imbalance and optimizes model performance for this important problem.


# In[1]:


# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_auc_score, roc_curve, classification_report


# In[2]:


# Load the dataset
df = pd.read_csv(r'C:\Users\apoly\OneDrive\Documents\GitHub\credit_card.csv\creditcard.csv')

# Basic data exploration
df.head()


# In[3]:


# ====================
# Data Preprocessing
# ====================
# Selecting features and target variables
features_numpy = df.iloc[:, 1:29]  # Features: V1 to V28
targets_numpy = df['Class'].values  # Target: Class (Fraud or Non-fraud)

# Normalizing the features (excluding the target)
scaler = StandardScaler()
features_numpy = scaler.fit_transform(features_numpy)

# Convert target values to numpy array and integers
targets_numpy = np.array(targets_numpy).astype(int)


# In[4]:


from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
features_resampled, targets_resampled = smote.fit_resample(features_numpy, targets_numpy)

# Then split the resampled dataset into training and testing sets
features_train, features_test, targets_train, targets_test = train_test_split(
    features_resampled, targets_resampled, test_size=0.1, random_state=42
)

print('features_train shape:', features_train.shape)
print('targets_train shape:', targets_train.shape)

print('features_test shape:', features_test.shape)
print('targets_test shape:', targets_test.shape)


# In[5]:


# ====================
# Convert to Torch Tensors
# ====================

# Convert to PyTorch tensors
featuresTrain = torch.tensor(features_train).float()
targetsTrain = torch.tensor(targets_train).long()
featuresTest = torch.tensor(features_test).float()
targetsTest = torch.tensor(targets_test).long()

# Create Datasets and DataLoaders
train_dataset = TensorDataset(featuresTrain, targetsTrain)
test_dataset = TensorDataset(featuresTest, targetsTest)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# In[6]:


# ====================
# Neural Network Model Definition
# ====================
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(28, 340)
        self.fc2 = nn.Linear(340, 220)
        self.fc3 = nn.Linear(220, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 2)  # Binary classification (fraud or non-fraud)
        self.dropout = nn.Dropout(0.3)  # Dropout to prevent overfitting
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.log_softmax(self.fc5(x), dim=1)
        return x
    
# Instantiate the model
model = Classifier()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[7]:


# ====================
# Training the Model
# ====================
num_epochs = 40
train_accuracy = []
loss_values = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Update training accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(trainloader)
    epoch_accuracy = 100 * correct / total
    loss_values.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


# In[8]:


# ====================
# Plot Training Metrics
# ====================

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_values, label='Train Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[9]:


# ====================
# Evaluation on Test Set
# ====================
model.eval()
with torch.no_grad():
    test_outputs = model(featuresTest)
    _, test_predictions = torch.max(test_outputs, 1)

test_acc = (test_predictions == targetsTest).sum().item() / targetsTest.size(0) * 100
print(f"Test Accuracy: {test_acc:.2f}%")

# Print classification report
print(classification_report(targetsTest, test_predictions.numpy(), target_names=['Non-Fraud', 'Fraud']))


# In[10]:


from sklearn.metrics import precision_recall_curve, average_precision_score

# Get the predicted probabilities for the test set
probs = F.softmax(test_outputs, dim=1)[:, 1].numpy()

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(targetsTest.numpy(), probs)

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Average precision score
avg_precision = average_precision_score(targetsTest.numpy(), probs)
print(f"Average Precision Score: {avg_precision:.2f}")


# In[11]:


# ====================
# ROC-AUC and Plot
# ====================
probabilities = F.softmax(test_outputs, dim=1)[:, 1].numpy()
roc_auc = roc_auc_score(targetsTest.numpy(), probabilities)
fpr, tpr, _ = roc_curve(targetsTest.numpy(), probabilities)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print(f"AUC-ROC: {roc_auc:.2f}")


# In[12]:


# Save the model
torch.save(model.state_dict(), 'fraud_detection_model.pth')

# Load the model
model = Classifier()
model.load_state_dict(torch.load('fraud_detection_model.pth'))
model.eval()

