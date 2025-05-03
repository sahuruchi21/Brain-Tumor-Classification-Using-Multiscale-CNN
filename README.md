
# 🧠 Brain Tumor Classification using Multiscale CNN

This project implements a deep learning model for classifying brain tumor types from MRI images using a custom-built Multiscale Convolutional Neural Network (CNN) in PyTorch.

## 🚀 Project Highlights

- Multiscale CNN architecture designed for effective feature extraction from medical images
- MRI image classification into tumor categories (e.g., glioma, meningioma, pituitary)
- Real-time training metrics and confusion matrix visualization
- Modular and readable PyTorch implementation

## 🗂️ Dataset

The dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/datasets) or from [Brain-Tumor-Classification-DataSet](https://github.com/).  
(📌 *Please ensure you have permission to use the dataset. Do not upload raw data to the repository.*)

## 🛠️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-classification.git
cd brain-tumor-classification
pip install -r requirements.txt
## 📦 Dataset

The dataset contains separate folders for training and testing MRI images of brain tumors.

Due to GitHub file size restrictions, the dataset ZIP file is not included in this repository.

📥 **Download the dataset** from [Google Drive](https://drive.google.com/file/d/1nWodPnBZTJvcKrkbBTeLf8oVFl8CMoW8/view?usp=drive_link) or the original [source repository](https://github.com/your-source-here).

After downloading:

1. Place the ZIP file in the project root directory.
2. The structure should look like this:
2. The structure should look like this:

brain-tumor-classification/
├── brain_tumor_dataset.zip              # Your ZIP file (downloaded manually)
├── data/                                # Extracted dataset folder (auto-generated)
│   └── Brain-Tumor-Classification-DataSet-master/
│       ├── Training/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   └── pituitary/
│       └── Testing/
│           ├── glioma/
│           ├── meningioma/
│           └── pituitary/
├── models/
│   └── multiscale_cnn.py
├── utils/
│   └── visualization.py
├── main.py
├── download_and_extract.py              # (Optional helper script)
├── requirements.txt
├── README.md
└── .gitignore

3. When you run the code, it will automatically extract the ZIP file to a folder and load the training/testing data.

