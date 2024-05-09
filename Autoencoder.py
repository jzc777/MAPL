import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_curve
from skimage import io
import os
import glob

class GaussianDistributionEstimator:
    def __init__(self, n_components=1, **kwargs):
        self.model = GaussianMixture(n_components=n_components, **kwargs)

    def fit(self, X):
        self.model.fit(X)

    def score_samples(self, X):
        return -self.model.score_samples(X)  # Returning the negative log-likelihood

def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(f'{folder}/*.jpg'):  # Adjust the path and file type as needed
        img = io.imread(filename)
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images, size=(256, 256)):
    processed_images = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Adjust as needed
    ])
    for img in images:
        processed_images.append(transform(img))
    return torch.stack(processed_images)

# Example usage
# Load and preprocess the dataset
normal_class_path = 'D:/python_work/AAAShenDuXueXi/Accf/MAPL/datasets/MVTec/capsule/train/good'  # Adjust this path
normal_images = load_images_from_folder(normal_class_path)
processed_images = preprocess_images(normal_images)
X = processed_images.view(processed_images.size(0), -1).numpy()  # Flatten the images

# Fit the Gaussian Distribution Estimators
pseudo_labeler = PseudoLabeler(n_occ_models=5, n_components=1)
pseudo_labeler.fit(X)

# Determine thresholds for pseudo-labeling
# Here, you need to implement logic to determine these thresholds based on your criteria, for example, using precision-recall curves

