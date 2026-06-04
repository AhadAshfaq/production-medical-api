import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

# Initialize FastAPI
app = FastAPI(
    title="Medical Image Inference API", 
    description="API for Pneumonia detection using ResNet18"
)

# 1. Define the model architecture exactly as in training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Loading API on device: {device}")

# We use weights=None because we are loading our own trained weights
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 2. Load the trained weights
# map_location ensures it loads safely regardless of hardware
model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval() # Set to evaluation mode!

# 3. Define the image preprocessing pipeline (Must match validation transforms)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['Normal', 'Pneumonia']

# 4. Define the inference endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read incoming image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess and prepare tensor
        # unsqueeze(0) adds the batch dimension required by PyTorch models
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # Inference without tracking gradients
        with torch.no_grad():
            outputs = model(input_tensor)
            # Apply softmax to convert raw logits into percentages
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)

        return {
            "prediction": classes[predicted_class.item()],
            "confidence": f"{round(confidence.item() * 100, 2)}%",
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))