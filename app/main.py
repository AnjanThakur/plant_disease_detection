import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import io
import uvicorn
import logging
import torch.nn as nn
from torchvision import models

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(BASE_DIR, "files", "best_tomato_model.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "files", "class_tomato_names.pt")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "files", "disease_info.csv")
SUPPLEMENT_INFO_PATH = os.path.join(BASE_DIR, "files", "supplement_info.csv")

# === MODEL DEFINITION ===
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Use the same model architecture
        self.features = models.resnet50(weights=None)  # We'll load weights from saved model
        
        in_features = self.features.fc.in_features
        self.features.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.features(x)
        
    def load_weights(self, model_path, device="cpu"):
        state_dict = torch.load(model_path, map_location=device)
        model_keys = list(state_dict.keys())

        if "model_state_dict" in model_keys:
            state_dict = state_dict["model_state_dict"]

        fc_weight_keys = [k for k in state_dict.keys() if "fc" in k and k.endswith(".weight")]
        if fc_weight_keys:
            loaded_fc_out = state_dict[fc_weight_keys[-1]].shape[0]
        expected_fc_out = self.features.fc[-1].out_features

        if expected_fc_out != loaded_fc_out:
            print(f"⚠️ Adjusting FC layer from {loaded_fc_out} to {expected_fc_out}")
            in_features = self.features.fc[0].in_features
            self.features.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, expected_fc_out)
            )
        else:
            print("No FC layer weights found in state_dict.")
        self.load_state_dict(state_dict)

# === MODEL AND DATA LOADING ===
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    device = get_device()
    try:
        # Check if files exist
        for file_path in [MODEL_PATH, CLASS_NAMES_PATH, DISEASE_INFO_PATH, SUPPLEMENT_INFO_PATH]:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        class_names = torch.load(CLASS_NAMES_PATH)
        num_classes = len(class_names)
        
        # Create model with the correct architecture
        model = CNN(num_classes=num_classes)
        
        # Use the custom load_weights method
        model.load_weights(MODEL_PATH, device=device)
        model.to(device)
        model.eval()
        
        disease_info = pd.read_csv(DISEASE_INFO_PATH)
        supplement_info = pd.read_csv(SUPPLEMENT_INFO_PATH)
        
        logger.info("Model and data loaded successfully")
        return model, class_names, disease_info, supplement_info, device
    
    except Exception as e:
        logger.error(f"Error loading model or data: {e}")
        raise

# Load model and data on first import
try:
    model, class_names, disease_info, supplement_info, device = load_model()
except Exception as e:
    logger.critical(f"Failed to load model or data: {e}")
    model, class_names, disease_info, supplement_info, device = None, None, None, None, None

# === IMAGE TRANSFORMATION ===
def transform_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# === PREDICTION FUNCTION ===
def predict_class(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# === DEPENDENCY INJECTION ===
def get_application_components():
    if model is None or class_names is None or disease_info is None or supplement_info is None or device is None:
        raise HTTPException(status_code=500, detail="Application not properly initialized - None component")
    if disease_info.empty or supplement_info.empty:
        raise HTTPException(status_code=500, detail="Application not properly initialized - Empty DataFrame")
    return {"model": model, "class_names": class_names,
            "disease_info": disease_info, "supplement_info": supplement_info}

# === FASTAPI APP ===
app = FastAPI(title="Tomato Disease Detection API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === API ROUTES ===
@app.get("/")
def home():
    return {"message": "Welcome to Tomato Disease Detection API 🚀",
            "status": "healthy" if model is not None else "not ready"}

@app.get("/health")
def health_check(components: dict = Depends(get_application_components)):
    return {"status": "healthy", "available_classes": len(components["class_names"])}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    components: dict = Depends(get_application_components)
):
    # Validate image format
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        # Read and process the image
        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
        
        # Transform and predict
        tensor = transform_image(image)
        pred_index = predict_class(tensor)
        
        # Get prediction results
        class_names = components["class_names"]
        disease_info = components["disease_info"]
        supplement_info = components["supplement_info"]
        
        if 0 <= pred_index < len(class_names):
            predicted_class_name = class_names[pred_index]
            
            # Get disease information
            disease_rows = disease_info[disease_info['disease_name'] == predicted_class_name]
            if disease_rows.empty:
                raise HTTPException(status_code=404, detail=f"Disease info not found for: {predicted_class_name}")
            
            disease_row = disease_rows.iloc[0]
            
            # Get supplement information
            supplement_rows = supplement_info[supplement_info['supplement name'] == disease_row['supplement name']]
            if supplement_rows.empty:
                raise HTTPException(status_code=404, detail=f"Supplement info not found for: {disease_row['supplement name']}")
            
            supplement_row = supplement_rows.iloc[0]
            
            # Prepare result
            result = {
                "predicted_class": predicted_class_name,
                "title": disease_row['disease_name'],
                "description": disease_row['description'],
                "prevent": disease_row['Possible Steps'],
                "image_url": disease_row['image_url'],
                "supplement_name": supplement_row['supplement name'],
                "supplement_image": supplement_row['supplement image'],
                "supplement_buy_link": supplement_row['buy link'],
            }
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail="Prediction index out of range")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# === RUN APP LOCALLY ===
if __name__ == "__main__":
  if model is None or class_names is None or disease_info is None or supplement_info is None:
      logger.critical("Cannot start application - initialization failed")
      exit(1)
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)