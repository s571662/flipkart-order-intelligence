import os
import argparse
import joblib
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

RETURN_MODEL_PATH = "models/return_risk_model.pkl"
THRESHOLD_PATH = "models/return_risk_threshold.txt"
IMAGE_MODEL_PATH = "models/product_classifier.pt"


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
    "Ankle boot"
]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("=" * 60)
    print("PART 3 - PREDICTION PIPELINE")
    print("=" * 60)

    print("\nChecking model files...")

    for path in [
        RETURN_MODEL_PATH,
        THRESHOLD_PATH,
        IMAGE_MODEL_PATH
    ]:
        if os.path.exists(path):
            print(f"OK: {path}")
        else:
            raise FileNotFoundError(
                f"Required model file not found: {path}"
            )

    # --------------------------------------------------------
    # Return risk model
    # --------------------------------------------------------

    print("\nLoading return risk model...")

    return_model = joblib.load(RETURN_MODEL_PATH)

    print("Return risk model loaded successfully.")

    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())

    print(f"Return risk threshold: {threshold:.4f}")

    # --------------------------------------------------------
    # Image model
    # --------------------------------------------------------

    print("\nLoading image classifier...")

    checkpoint = torch.load(
        IMAGE_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    print("Image classifier file loaded successfully.")

    print("\nCreating ResNet-18 architecture...")

    image_model = models.resnet18(weights=None)

    image_model.fc = nn.Linear(
        image_model.fc.in_features,
        10
    )

    image_model.load_state_dict(checkpoint)

    image_model.eval()

    print("ResNet-18 architecture created.")
    print("Trained weights loaded.")

    print("\n" + "=" * 60)
    print("PART 3 MODEL LOADING SUCCESSFUL")
    print("=" * 60)

    return return_model, threshold, image_model


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_product_image(image_model, image_path):

    print("\n" + "=" * 60)
    print("IMAGE CLASSIFICATION")
    print("=" * 60)

    print(f"Image: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("L")

    print(f"Original image size: {image.size}")

    image_tensor = image_transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    print(f"Model input shape: {image_tensor.shape}")

    with torch.no_grad():

        outputs = image_model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    class_name = CLASS_NAMES[predicted_class]

    print(f"Predicted class: {predicted_class}")
    print(f"Predicted product: {class_name}")
    print(f"Confidence: {confidence:.4f}")

    return {
        "class_id": predicted_class,
        "product_class": class_name,
        "confidence": confidence
    }


# ============================================================
# RETURN RISK PREDICTION
# ============================================================

def predict_return_risk(
    return_model,
    threshold,
    order_data
):

    print("\n" + "=" * 60)
    print("RETURN RISK PREDICTION")
    print("=" * 60)

    order_df = pd.DataFrame([order_data])

    print("\nOrder data:")
    print(order_df.to_string(index=False))

    return_probability = (
        return_model
        .predict_proba(order_df)[0, 1]
    )

    return_prediction = int(
        return_probability >= threshold
    )

    if return_prediction == 1:
        risk_label = "HIGH RETURN RISK"
    else:
        risk_label = "LOW RETURN RISK"

    print("\nReturn probability:")
    print(f"{return_probability:.4f}")

    print("\nThreshold:")
    print(f"{threshold:.4f}")

    print("\nPrediction:")
    print(risk_label)

    return {
        "return_probability": return_probability,
        "return_prediction": return_prediction,
        "risk_label": risk_label
    }


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Flipkart Order Intelligence Prediction Pipeline"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to product image"
    )

    parser.add_argument(
        "--price",
        type=float,
        required=True
    )

    parser.add_argument(
        "--distance",
        type=float,
        required=True
    )

    parser.add_argument(
        "--tenure",
        type=int,
        required=True
    )

    parser.add_argument(
        "--delivery-days",
        type=int,
        required=True
    )

    parser.add_argument(
        "--discount",
        type=float,
        required=True
    )

    parser.add_argument(
        "--previous-orders",
        type=int,
        required=True
    )

    parser.add_argument(
        "--previous-returns",
        type=int,
        required=True
    )

    parser.add_argument(
        "--rating",
        type=float,
        required=True
    )

    parser.add_argument(
        "--payment",
        required=True
    )

    parser.add_argument(
        "--category",
        required=True
    )

    parser.add_argument(
        "--weekend",
        type=int,
        choices=[0, 1],
        required=True
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    return_model, threshold, image_model = load_models()

    # --------------------------------------------------------
    # Image prediction
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RUNNING IMAGE PREDICTION")
    print("=" * 60)

    image_result = predict_product_image(
        image_model,
        args.image
    )

    # --------------------------------------------------------
    # Order data
    # --------------------------------------------------------

    order_data = {
        "price_inr": args.price,
        "delivery_distance_km": args.distance,
        "customer_tenure_days": args.tenure,
        "delivery_days": args.delivery_days,
        "discount_pct": args.discount,
        "num_previous_orders": args.previous_orders,
        "num_previous_returns": args.previous_returns,
        "rating_given": args.rating,
        "payment_method": args.payment,
        "product_category": args.category,
        "is_weekend_order": args.weekend
    }

    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return_result = predict_return_risk(
        return_model,
        threshold,
        order_data
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL ORDER INTELLIGENCE")
    print("=" * 60)

    print(
        f"Product:             "
        f"{image_result['product_class']}"
    )

    print(
        f"Image confidence:    "
        f"{image_result['confidence']:.2%}"
    )

    print(
        f"Return probability:  "
        f"{return_result['return_probability']:.2%}"
    )

    print(
        f"Return risk:         "
        f"{return_result['risk_label']}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()