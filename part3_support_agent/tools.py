from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

RETURN_MODEL_PATH = BASE_DIR / "models" / "return_risk_model.pkl"
RETURN_THRESHOLD_PATH = BASE_DIR / "models" / "return_risk_threshold.txt"


def check_return_risk(order_features: dict) -> dict:
    model = joblib.load(RETURN_MODEL_PATH)

    with open(RETURN_THRESHOLD_PATH, "r") as f:
        t_rf = float(f.read().strip())

    order_df = pd.DataFrame([order_features])

    probability = float(
        model.predict_proba(order_df)[0, 1]
    )

    if probability < t_rf:
        risk_bucket = "Low"
    elif probability >= t_rf + 0.15:
        risk_bucket = "High"
    else:
        risk_bucket = "Medium"

    return {
        "return_probability": probability,
        "risk_bucket": risk_bucket,
        "threshold": t_rf,
    }

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


IMAGE_MODEL_PATH = BASE_DIR / "models" / "product_classifier.pt"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def classify_product_image(image_path: str) -> dict:
    image_model = models.resnet18(weights=None)

    image_model.fc = nn.Linear(
        image_model.fc.in_features,
        10,
    )

    checkpoint = torch.load(
        IMAGE_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    image_model.load_state_dict(checkpoint)
    image_model.eval()

    image = Image.open(image_path).convert("L")
    image_tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = image_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = int(
            torch.argmax(probabilities, dim=1).item()
        )

        confidence = float(
            probabilities[0, predicted_class].item()
        )

    return {
        "class_id": predicted_class,
        "product_class": CLASS_NAMES[predicted_class],
        "confidence": confidence,
    }