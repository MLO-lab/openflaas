import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.colors import ListedColormap, Normalize
from sklearn.manifold import TSNE


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    accuracy = correct / len(loader.dataset)
    return running_loss / len(loader.dataset), accuracy


def get_client_features(dataloader, model, device):

    model.fc = nn.Identity()
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            output = model(images)  # Get penultimate layer features
            features.append(output.cpu())
            labels.extend(lbls.numpy())

    features = torch.cat(features).numpy()
    labels = np.array(labels)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(features)

    return embeddings_2d, labels


def plot_client_features(embeddings_2d, labels, score_idx):
    cmap = ListedColormap(cm.tab10.colors[:10])
    norm = Normalize(vmin=0, vmax=9)
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    most_uncertain_points = embeddings_2d[score_idx.tolist()]
    most_uncertain_labels = labels[score_idx.tolist()]

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.7, s=10)

    plt.scatter(
        most_uncertain_points[:, 0], most_uncertain_points[:, 1],
        c=most_uncertain_labels, cmap=cmap, norm=norm, edgecolor='black', s=50, label=f"Most uncertain points"
    )

    plt.legend()
    plt.show()