import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.colors import ListedColormap, Normalize
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from torch import Tensor


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


def plot_client_features(embeddings_2d, labels, score_idx, n_classes=10):
    cmap = ListedColormap(cm.tab20.colors)
    norm = Normalize(vmin=0, vmax=n_classes - 1)
    
    most_uncertain_points = embeddings_2d[score_idx.tolist()]
    most_uncertain_labels = labels[score_idx.tolist()]

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.7, s=10)

    plt.scatter(
        most_uncertain_points[:, 0], most_uncertain_points[:, 1],
        c=most_uncertain_labels, cmap=cmap, norm=norm, edgecolor='black', s=50, label=f"Most uncertain points"
    )

    plt.legend()
    plt.show()


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def generate_embedding(text_list):
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
    model = AutoModel.from_pretrained('intfloat/e5-small-v2')
    
    output_list = []
    model.eval()
    for i in tqdm(range(0, len(text_list), 10)):
        chunk = text_list[i:i+10]
        batch_dict = tokenizer(chunk, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        output_list.append(embeddings)

    output = torch.cat(output_list)
    return output


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x