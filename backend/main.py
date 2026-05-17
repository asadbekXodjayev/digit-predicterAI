import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# Define the CNN architecture matching the training model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Fully connected layers (64 * 7 * 7 = 3136 after 2 pooling operations)
        # Input: 28x28 → after pool1: 14x14 → after pool2: 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # First conv block (28x28 → 14x14)
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block (14x14 → 7x7)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and fully connected (7x7 * 64 = 3136)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout3(nn.functional.relu(self.bn5(self.fc1(x))))
        x = self.fc2(x)
        
        return x

# Pydantic model for request validation
class PixelData(BaseModel):
    pixels: List[float] = Field(
        ..., 
        min_items=784, 
        max_items=784,
        description="Flat array of 784 pixel values normalized between 0 and 1"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "pixels": [0.0] * 784
            }
        }

# Initialize FastAPI app
app = FastAPI(
    title="Hand-Drawn Digit Recognizer API",
    description="API for recognizing hand-drawn digits using a trained CNN neural network",
    version="2.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = None

def load_model():
    """Load the trained model at startup"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    
    try:
        model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
        model.eval()
        print(f"CNN Model loaded successfully on device: {device}")
        return True
    except FileNotFoundError:
        print("Warning: mnist_model.pth not found. Run train.py first to train the model.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Hand-Drawn Digit Recognizer API (CNN v2.0)",
        "status": "running",
        "endpoints": {
            "/predict-digit": "POST - Submit 784 pixel values for digit prediction",
            "/health": "GET - Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and model.training == False
    }

@app.post("/predict-digit")
async def predict_digit(pixel_data: PixelData):
    """
    Predict the digit from 784 pixel values.
    
    The input should be a flat array of 784 values (28x28 image flattened),
    with each value normalized between 0 and 1.
    
    Returns the predicted digit (0-9) and confidence scores for all digits.
    """
    global model, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert input to tensor and reshape
        pixels = np.array(pixel_data.pixels, dtype=np.float32)
        
        # Reshape from (784,) to (1, 1, 28, 28)
        # MNIST expects grayscale images with shape (batch, channels, height, width)
        image = pixels.reshape(1, 1, 28, 28)
        
        # Convert to tensor
        tensor = torch.from_numpy(image).to(device)
        
        # Apply the SAME normalization used during training
        # MNIST normalization: mean=0.1307, std=0.3081
        mean = 0.1307
        std = 0.3081
        tensor = (tensor - mean) / std
        
        # Perform prediction
        model.eval()
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get confidence scores for all digits
        confidence_scores = {str(i): float(probabilities[0][i].item()) for i in range(10)}
        
        return {
            "predicted_digit": int(predicted.item()),
            "confidence": float(confidence.item()),
            "confidence_scores": confidence_scores
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)