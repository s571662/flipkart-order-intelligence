import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
print("PyTorch version:", torch.__version__)

data_dir = "part2_image_classifier/data"

print("\nDownloading/loading Fashion-MNIST...")

train_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=True,
    download=True
)

test_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=False,
    download=True
)

print("\nDataset loaded successfully!")
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))

print("\nFirst training image:")

image, label = train_dataset[0]

print("Image type:", type(image))
print("Image size:", image.size)
print("Label:", label)

class_names = [
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

# ---------------------------------------------------------
# ResNet-18 preprocessing
# ---------------------------------------------------------

image_size = 224

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("\nResNet preprocessing configured:")
print("Input size:", image_size, "x", image_size)
print("Channels: 3")
print("Normalization: ImageNet mean/std")

print("\nFashion-MNIST classes:")

for index, name in enumerate(class_names):
    print(index, "=", name)
    
    
from sklearn.model_selection import train_test_split
# ---------------------------------------------------------
# Export real test-split sample images for Part 3
# ---------------------------------------------------------

os.makedirs("data/sample_images", exist_ok=True)

sample_indices = [0, 1, 2, 3, 4]

print("\nExporting sample test images...")

for idx in sample_indices:
    image, label = test_dataset[idx]

    safe_name = class_names[label].lower().replace("/", "_").replace(" ", "_")

    output_path = f"data/sample_images/{idx:02d}_{safe_name}.png"

    image.save(output_path)

    print(f"Saved: {output_path} | True label: {class_names[label]}")
# ---------------------------------------------------------
# Stratified train/validation split
# ---------------------------------------------------------

all_labels = train_dataset.targets.numpy()

train_indices, val_indices = train_test_split(
    range(len(train_dataset)),
    test_size=5000,
    random_state=42,
    stratify=all_labels
)

print("\n" + "=" * 60)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

print("Original training images:", len(train_dataset))
print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))
print("Test images:", len(test_dataset))

# ---------------------------------------------------------
# Test preprocessing on one image
# ---------------------------------------------------------

sample_image, sample_label = train_dataset[0]

transformed_image = transform(sample_image)

print("\nTransformed sample:")
print("Original size:", sample_image.size)
print("Transformed shape:", transformed_image.shape)
print("Label:", sample_label)

# ---------------------------------------------------------
# Load pretrained ResNet-18
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING PRETRAINED RESNET-18")
print("=" * 60)

weights = models.ResNet18_Weights.DEFAULT

resnet = models.resnet18(weights=weights)

print("Pretrained ResNet-18 loaded successfully.")

# Freeze the entire pretrained backbone
for param in resnet.parameters():
    param.requires_grad = False

# Get the number of input features to the original classifier
num_features = resnet.fc.in_features

# Replace the ImageNet classifier with our 10-class classifier
resnet.fc = torch.nn.Linear(
    num_features,
    10
)

print("Original classifier replaced.")
print("Classifier input features:", num_features)
print("Classifier output classes:", 10)

# Count trainable parameters
trainable_parameters = sum(
    p.numel()
    for p in resnet.parameters()
    if p.requires_grad
)

total_parameters = sum(
    p.numel()
    for p in resnet.parameters()
)

print("\nTotal parameters:", total_parameters)
print("Trainable parameters:", trainable_parameters)
print("Frozen parameters:", total_parameters - trainable_parameters)

# ---------------------------------------------------------
# Create datasets using the ResNet preprocessing
# ---------------------------------------------------------

train_dataset_transformed = datasets.FashionMNIST(
    root="part2_image_classifier/data",
    train=True,
    download=True,
    transform=transform
)

test_dataset_transformed = datasets.FashionMNIST(
    root="part2_image_classifier/data",
    train=False,
    download=True,
    transform=transform
)

# Same split used earlier:
# 55,000 training
# 5,000 validation

train_indices, val_indices = train_test_split(
    range(len(train_dataset_transformed)),
    test_size=5000,
    stratify=train_dataset_transformed.targets,
    random_state=42
)

train_subset = torch.utils.data.Subset(
    train_dataset_transformed,
    train_indices
)

val_subset = torch.utils.data.Subset(
    train_dataset_transformed,
    val_indices
)

print("\nDatasets prepared:")
print("Training:", len(train_subset))
print("Validation:", len(val_subset))
print("Test:", len(test_dataset_transformed))

# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------

batch_size = 64

train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset_transformed,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print("\nDataLoaders created.")
print("Batch size:", batch_size)

# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)

resnet = resnet.to(device)
resnet.eval()


def extract_features(model, loader, device):
    """
    Run images through the frozen ResNet-18 and
    return the 512-dimensional feature vectors.
    """

    features = []
    labels = []

    with torch.no_grad():

        for batch_images, batch_labels in loader:

            batch_images = batch_images.to(device)

            # ResNet layers except the final classifier
            x = model.conv1(batch_images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)

            x = torch.flatten(x, 1)

            features.append(x.cpu())
            labels.append(batch_labels)

    features = torch.cat(features)
    labels = torch.cat(labels)

    return features, labels
# ---------------------------------------------------------
# Extract cached features
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EXTRACTING TRAINING FEATURES")
print("=" * 60)

train_features, train_labels = extract_features(
    resnet,
    train_loader,
    device
)

print("Training feature shape:", train_features.shape)

print("\n" + "=" * 60)
print("EXTRACTING VALIDATION FEATURES")
print("=" * 60)

val_features, val_labels = extract_features(
    resnet,
    val_loader,
    device
)

print("Validation feature shape:", val_features.shape)

print("\n" + "=" * 60)
print("EXTRACTING TEST FEATURES")
print("=" * 60)

test_features, test_labels = extract_features(
    resnet,
    test_loader,
    device
)

print("Test feature shape:", test_features.shape)

# ---------------------------------------------------------
# Train classifier head on cached ResNet features
# ---------------------------------------------------------

from torch.utils.data import TensorDataset

print("\n" + "=" * 60)
print("TRAINING CLASSIFIER HEAD")
print("=" * 60)

head_train_dataset = TensorDataset(
    train_features,
    train_labels
)

head_val_dataset = TensorDataset(
    val_features,
    val_labels
)

head_train_loader = DataLoader(
    head_train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

head_val_loader = DataLoader(
    head_val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

# Create a fresh classifier head
classifier = torch.nn.Linear(512, 10)

classifier = classifier.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=0.001
)

epochs = 10

print("Optimizer: Adam")
print("Learning rate: 0.001")
print("Batch size: 128")
print("Epochs:", epochs)

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

best_val_accuracy = 0.0
best_classifier_state = None

for epoch in range(epochs):

    classifier.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in head_train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = classifier(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * features.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    classifier.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for features, labels in head_val_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = classifier(features)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # Save best validation model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_classifier_state = {
            key: value.cpu().clone()
            for key, value in classifier.state_dict().items()
        }

print("\nBest validation accuracy:")
print(f"{best_val_accuracy:.4f}")

# ---------------------------------------------------------
# Restore best validation classifier
# ---------------------------------------------------------

classifier.load_state_dict(best_classifier_state)

classifier = classifier.to(device)

classifier.eval()

print("\nBest classifier restored.")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")

# ---------------------------------------------------------
# Test-set predictions
# ---------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

test_predictions = []
test_actual = []

with torch.no_grad():

    for features, labels in DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=128,
        shuffle=False
    ):

        features = features.to(device)

        outputs = classifier(features)

        predictions = outputs.argmax(dim=1)

        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_actual.extend(
            labels.numpy()
        )

test_predictions = np.array(test_predictions)
test_actual = np.array(test_actual)

test_accuracy = accuracy_score(
    test_actual,
    test_predictions
)

print("\n" + "=" * 60)
print("FINAL TEST-SET RESULTS")
print("=" * 60)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

class_names = [
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

cm = confusion_matrix(
    test_actual,
    test_predictions
)

print("\n" + "=" * 60)
print("10 x 10 CONFUSION MATRIX")
print("=" * 60)

print(cm)

# ---------------------------------------------------------
# Per-class precision and recall
# ---------------------------------------------------------

precision, recall, f1, support = precision_recall_fscore_support(
    test_actual,
    test_predictions,
    labels=np.arange(10),
    zero_division=0
)

print("\n" + "=" * 60)
print("PER-CLASS PRECISION / RECALL")
print("=" * 60)

for i, class_name in enumerate(class_names):

    print(
        f"{class_name:15s} | "
        f"Precision: {precision[i]:.4f} | "
        f"Recall: {recall[i]:.4f} | "
        f"F1: {f1[i]:.4f} | "
        f"Support: {support[i]}"
    )
    
# ---------------------------------------------------------
# Most common confusion pairs
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MOST COMMON CONFUSION PAIRS")
print("=" * 60)

confusion_pairs = []

for actual_class in range(10):

    for predicted_class in range(10):

        if actual_class == predicted_class:
            continue

        confusion_pairs.append(
            (
                cm[actual_class, predicted_class],
                class_names[actual_class],
                class_names[predicted_class]
            )
        )

confusion_pairs.sort(
    reverse=True,
    key=lambda x: x[0]
)

for count, actual_name, predicted_name in confusion_pairs[:10]:

    print(
        f"Actual {actual_name} -> "
        f"Predicted {predicted_name}: "
        f"{count}"
    )
    
# ---------------------------------------------------------
# Save final complete ResNet-18 classifier
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SAVING PRODUCT CLASSIFIER")
print("=" * 60)

# Put trained classifier weights into ResNet's final layer
resnet.fc.load_state_dict(
    classifier.state_dict()
)

# Move complete model to CPU
resnet = resnet.cpu()

# Evaluation mode
resnet.eval()

# Create models directory
os.makedirs("models", exist_ok=True)

model_path = "models/product_classifier.pt"

torch.save(
    resnet.state_dict(),
    model_path
)

print("Model saved successfully:")
print(model_path)

print("\n============================================================")
print("MODEL SAVED")
print("============================================================")
print(f"Saved to: {model_path}")