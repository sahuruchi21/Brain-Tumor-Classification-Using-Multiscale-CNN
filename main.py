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
        image = Image.open(self.image_paths[idx]).convert('RGB')
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
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

        train_acc = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds, average='macro')
        train_recall = recall_score(all_labels, all_preds, average='macro')
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        metrics['accuracy'].append(train_acc)
        metrics['precision'].append(train_precision)
        metrics['recall'].append(train_recall)
        metrics['f1_score'].append(train_f1)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Acc: {train_acc:.4f} - Precision: {train_precision:.4f} - Recall: {train_recall:.4f} - F1-score: {train_f1:.4f}")

    return model

# Function to Evaluate Model on Test Dataset
def evaluate_model(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')
    test_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n🔹 Test Accuracy: {test_acc:.4f}")
    print(f"🔹 Test Precision: {test_precision:.4f}")
    print(f"🔹 Test Recall: {test_recall:.4f}")
    print(f"🔹 Test F1-score: {test_f1:.4f}")

    test_conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 10})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix - Test Data', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

    return all_preds

# Function to Display Sample Predictions
def display_sample_predictions(model, dataset, class_names, num_images=5):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fig = plt.figure(figsize=(num_images * 3, 4))

    for i in range(num_images):
        image, label = dataset[i]
        input_img = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)
            _, pred = torch.max(output, 1)

        ax = fig.add_subplot(1, num_images, i+1)
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
        img_np = np.clip(img_np, 0, 1)

        ax.imshow(img_np)
        ax.set_title(f"Actual:\n{class_names[label]}\nPred:\n{class_names[pred.item()]}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Run Training and Evaluation
model = MultiscaleCNN()
trained_model = train_model(model, train_loader, val_loader, epochs=30)

# Evaluate on Test Set
evaluate_model(trained_model, test_loader)

# Show Sample Predictions
display_sample_predictions(trained_model, test_dataset, class_names, num_images=5)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Grad-CAM generation function
def generate_gradcam(model, image_tensor, target_layer):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0)
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    class_score = output[0, pred_class]
    model.zero_grad()
    class_score.backward()

    activation = activations[0]
    gradient = gradients[0]

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1).squeeze()
    cam = torch.clamp(cam, min=0)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam =  cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (128, 128))

    handle_forward.remove()
    handle_backward.remove()

    return cam

# Grad-CAM visualization with bounding boxes
def display_predictions_with_gradcam(model, dataset, class_names, target_layer, num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    indices = np.random.choice(len(dataset), num_images, replace=False)

    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image_tensor = image.to(device)

        gradcam = generate_gradcam(model, image_tensor, target_layer)

        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image_np = np.clip(image_np, 0, 1)

        heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap.astype(np.uint8), cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        overlay = 0.4 * heatmap + 0.6 * image_np
        overlay = np.clip(overlay, 0, 1)

        thresh = gradcam > gradcam.mean()
        coords = np.argwhere(thresh)

        if coords.size != 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            overlay_uint8 = (overlay * 255).astype(np.uint8)
            overlay_box = cv2.rectangle(overlay_uint8.copy(), (x0, y0), (x1, y1), (0, 255, 0), 2)
            overlay_box = overlay_box.astype(np.float32) / 255.0
        else:
            overlay_box = overlay

        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0)).argmax(dim=1).item()

        plt.subplot(1, num_images, i + 1)
        plt.imshow(overlay_box)
        plt.title(f"Actual: {class_names[label]}\nPred: {class_names[pred]}", fontsize=9)
        plt.axis('off')

    plt.suptitle("Grad-CAM with Bounding Boxes", fontsize=14)
    plt.tight_layout()
    plt.show()

# Choose the target layer (last conv layer of your model)
target_layer = trained_model.conv3

# Display Grad-CAM results
display_predictions_with_gradcam(trained_model, val_dataset, class_names, target_layer, num_images=10)


