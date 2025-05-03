import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from PIL import Image
import math

# Extract dataset
zip_path = "/content/Brain-Tumor-Classification-DataSet-master.zip"
unzip_path = "/content/brain_tumor_data"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_path)

# Define dataset paths
dataset_path = os.path.join(unzip_path, "Brain-Tumor-Classification-DataSet-master", "Training")
test_dataset_path = os.path.join(unzip_path, "Brain-Tumor-Classification-DataSet-master", "Testing")

# Get class labels dynamically
class_names = sorted(os.listdir(dataset_path))
num_classes = len(class_names)
print(f"Number of Classes Detected: {num_classes}")

# Dataset Class
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label, tumor_type in enumerate(os.listdir(root_dir)):
            tumor_folder = os.path.join(root_dir, tumor_type)
            if os.path.isdir(tumor_folder):
                for img_name in os.listdir(tumor_folder):
                    img_path = os.path.join(tumor_folder, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Use RGB instead of Grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Train and Test Datasets
dataset = BrainTumorDataset(dataset_path, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

test_dataset = BrainTumorDataset(test_dataset_path, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Function to Display Images with Labels
def display_images(dataset, title, num_images=5):
    plt.figure(figsize=(10, 5))
    indices = np.random.choice(len(dataset), num_images, replace=False)

    for i, idx in enumerate(indices):
        image, label = dataset[idx]  # Get image and label
        image = image.permute(1, 2, 0).numpy()  # Convert to NumPy format
        image = (image * 0.229) + 0.485  # Reverse normalization for display

        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        plt.title(class_names[label], fontsize=12)
        plt.axis('off')

    plt.suptitle(title, fontsize=14)
    plt.show()

# Display Train and Test Images
display_images(train_dataset, "Sample Train Images", num_images=5)
display_images(test_dataset, "Sample Test Images", num_images=5)

# Define Multiscale CNN Model
class MultiscaleCNN(nn.Module):
    def __init__(self):
        super(MultiscaleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Train & Evaluate Model
def train_model(model, train_loader, val_loader, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute Metrics
        train_acc = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds, average='macro')
        train_recall = recall_score(all_labels, all_preds, average='macro')
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        metrics['accuracy'].append(train_acc)
        metrics['precision'].append(train_precision)
        metrics['recall'].append(train_recall)
        metrics['f1_score'].append(train_f1)

        # Confusion Matrix
        train_conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Acc: {train_acc:.4f} - Precision: {train_precision:.4f} - Recall: {train_recall:.4f} - F1-score: {train_f1:.4f}")

        plt.figure(figsize=(6, 5))
        sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.show()

    # Plot Performance Metrics
    plt.figure(figsize=(12, 5))
    for metric, values in metrics.items():
        plt.plot(range(1, epochs + 1), values, label=metric)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Training Metrics Over Epochs")
    plt.show()

    return model

# Run Training
model = MultiscaleCNN()
trained_model = train_model(model, train_loader, val_loader, epochs=30)


