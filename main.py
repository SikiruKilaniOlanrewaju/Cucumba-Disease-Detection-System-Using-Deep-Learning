print("Hello, World! This is the Cucumba Disease Detection System Using Deep Learning.")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch is not installed. Please install it using: pip install torch")

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch.optim as optim  # <-- Fix: Import torch.optim

# Example: Simple neural network for image classification
class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(128 * 128 * 3, 256)  # Adjusted for 128x128 RGB images
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the path to your dataset directory after unzipping
DATASET_PATH = os.path.join(os.getcwd(), 'Cucumber Plant Diseases Dataset')

# Define image transformations (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Set the path to your training dataset directory
TRAIN_PATH = os.path.join(DATASET_PATH, 'training')

# Load only the training set (good_Cucumber and Ill_cucumber subfolders)
train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_classes = len(train_dataset.classes)

# Instantiate and print the model with correct number of classes
model = SimpleNet(num_classes)
print(model)

# Print info about the training dataset
print(f"Total training images: {len(train_dataset)}")
print(f"Training classes: {train_dataset.classes}")

# Example: Get a batch of images and labels from the training set
images, labels = next(iter(train_loader))
print(f"Training batch shape: {images.shape}")
print(f"Training batch labels: {labels}")

# Training loop example

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5  # You can increase this for better results
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), -1)  # Flatten images for SimpleNet
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Validation after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy after epoch {epoch+1}: {accuracy:.2f}%")

# Save the trained model
model_save_path = os.path.join(os.getcwd(), 'cucumber_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# --- TESTING/VALIDATION DATASET LOADING ---
TEST_PATH = os.path.join(DATASET_PATH, 'testing')
test_dataset = datasets.ImageFolder(root=TEST_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Total test images: {len(test_dataset)}")
print(f"Test classes: {test_dataset.classes}")

# --- FINAL EVALUATION ON TEST SET ---
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

# Final evaluation on test set
final_test_acc = evaluate(model, test_loader, device)
print(f"Final Test Accuracy: {final_test_acc:.2f}%")

# --- INFERENCE FUNCTION FOR SINGLE IMAGE ---
from PIL import Image

def predict_image(image_path, model, device, transform, class_names):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    image = image.view(image.size(0), -1)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Example usage for single prediction
def single_prediction_example():
    single_pred_dir = os.path.join(DATASET_PATH, 'single_prediction')
    for fname in os.listdir(single_pred_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(single_pred_dir, fname)
            pred = predict_image(img_path, model, device, transform, train_dataset.classes)
            print(f"Image: {fname} -> Predicted: {pred}")

# Uncomment to run single prediction example after training
# single_prediction_example()
