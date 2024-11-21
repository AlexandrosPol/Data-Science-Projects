# Emotional Image Classification

## Overview
This project is a **deep learning-based web application** designed to predict the emotion conveyed in an image as either **happy** or **sad**. The aim is to demonstrate the practical application of convolutional neural networks (CNNs) in emotion recognition through a user-friendly API.

Users can upload images, and the model provides predictions with high accuracy.

---

## Key Features
- **Deep Learning Model**: Built using MobileNetV3 architecture for efficient and accurate emotion classification.
- **User Interaction**: API endpoints for uploading images and receiving predictions.
- **Batch Processing**: Supports single and batch image predictions, including ZIP file uploads.

---

## Deployment
The application is deployed on **Heroku** and is accessible at:  
[Emotion Classification API](https://emotion-classifier-2024.herokuapp.com/)

---

## How to Use
1. **Root Endpoint**: Visit the root URL to see a welcome message.
2. **Swagger UI**: Use the interactive API documentation at `/docs` to test the endpoints.
3. **Endpoints**:
   - `/predict`: Upload a single image for classification.
   - `/predict-batch`: Upload multiple images for batch predictions.
   - `/predict-batch-zip`: Upload a ZIP file containing multiple images for classification.

---

## Technology Stack
- **TensorFlow**: For building and training the CNN model.
- **FastAPI**: For creating the API.
- **Heroku**: For deployment and hosting.
- **Docker**: For containerizing the application.

---

## Future Enhancements
- Extend emotion categories to cover a broader range of feelings.
- Improve the API response time for large-scale predictions.
- Create a graphical user interface for end-users.

---

## Acknowledgments
This project aims to showcase the use of AI in emotion recognition, inspired by the growing interest in mental health analytics.