import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from scipy.stats import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

digits = load_digits()

X = digits.images
y = digits.target

X = X / 16.0

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256
)

class SimpleCNN(nn.Module):

    def __init__(self, num_filters=32, dropout=0.3):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            num_filters,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            num_filters,
            num_filters * 2,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2)

        # 8x8 -> 4x4 -> 2x2
        self.fc1 = nn.Linear(
            (num_filters * 2) * 2 * 2,
            128
        )

        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))

        x = self.fc2(x)

        return x


visual_similarity = {

    (1, 7): 0.90,
    (3, 8): 0.85,
    (5, 6): 0.80,
    (0, 6): 0.75,
    (2, 7): 0.70,
    (4, 9): 0.65,
}

def get_visual_similarity(a, b):

    if (a, b) in visual_similarity:
        return visual_similarity[(a, b)]

    if (b, a) in visual_similarity:
        return visual_similarity[(b, a)]

    return 0.1


def train_model(params):

    lr = params["lr"]
    dropout = params["dropout"]
    filters = params["filters"]

    model = SimpleCNN(
        num_filters=filters,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    EPOCHS = 5

    for epoch in range(EPOCHS):

        total_loss = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}, "
            f"Loss = {total_loss:.4f}"
        )

    return model


def build_confusion_matrix(model):

    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)

            logits = model(x)

            pred = torch.argmax(logits, dim=1)

            all_true.extend(y.numpy())
            all_pred.extend(pred.cpu().numpy())

    cm = confusion_matrix(all_true, all_pred)

    return cm, np.array(all_true), np.array(all_pred)


def compute_confusion_sensitivity(
    cm,
    alpha=0.5
):

    n = cm.shape[0]

    sensitivity = {}

    for a in range(n):

        for b in range(a + 1, n):

            confusion_ab = cm[a][b]
            confusion_ba = cm[b][a]

            confusion_degree = (
                confusion_ab + confusion_ba
            ) / 2

            visual_score = get_visual_similarity(a, b)

            score = (
                (1 - alpha) * confusion_degree
                + alpha * visual_score
            )

            sensitivity[(a, b)] = score

    return sensitivity


def select_high_confusion_pairs(
    sensitivity,
    K=5
):

    sorted_pairs = sorted(
        sensitivity.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_pairs = sorted_pairs[:K]

    total = sum(v for _, v in top_pairs)

    pair_weights = {}

    for pair, value in top_pairs:

        pair_weights[pair] = value / total

    return pair_weights


def confusion_weighted_accuracy(
    y_true,
    y_pred,
    pair_weights,
    base_weight=1.0
):

    numerator = 0
    denominator = 0

    for t, p in zip(y_true, y_pred):

        pair = tuple(sorted((int(t), int(p))))

        confusion_weight = pair_weights.get(pair, 0)

        importance_weight = (
            base_weight + confusion_weight
        )

        correct = 1 if t == p else 0

        numerator += (
            importance_weight * correct
        )

        denominator += importance_weight

    return numerator / denominator


def expected_improvement(
    mu,
    sigma,
    best
):

    if sigma < 1e-9:
        return 0

    z = (mu - best) / sigma

    ei = (
        (mu - best) * norm.cdf(z)
        + sigma * norm.pdf(z)
    )

    return ei

# =========================================================
# CONFUSION UNCERTAINTY TERM
# 专利创新点
# =========================================================

def confusion_uncertainty_term(
    sigma,
    pair_weights,
    persistent_pairs
):

    value = 0

    for pair in persistent_pairs:

        weight = pair_weights.get(pair, 0)

        value += weight * sigma

    return value


def confusion_aware_acquisition(
    mu,
    sigma,
    best_score,
    pair_weights,
    persistent_pairs,
    beta=0.5
):

    ei = expected_improvement(
        mu,
        sigma,
        best_score
    )

    confusion_term = confusion_uncertainty_term(
        sigma,
        pair_weights,
        persistent_pairs
    )

    acquisition = (
        ei + beta * confusion_term
    )

    return acquisition


def sample_hyperparameters():

    return {

        "lr": 10 ** random.uniform(-4, -2),

        "dropout": random.uniform(0.2, 0.6),

        "filters": random.choice([16, 32, 64])
    }


def params_to_vector(params):

    return np.array([

        params["lr"],

        params["dropout"],

        params["filters"]
    ])


history_X = []
history_y = []

best_score = -1
best_params = None

persistent_pairs = []

N_ITER = 10

for iteration in range(N_ITER):

    print("\n")
    print("=" * 50)
    print(f"ITERATION {iteration+1}")
    print("=" * 50)

    params = sample_hyperparameters()

    print("Hyperparameters:")
    print(params)

    model = train_model(params)

    cm, y_true, y_pred = build_confusion_matrix(model)

    print("\nConfusion Matrix:")
    print(cm)

    sensitivity = compute_confusion_sensitivity(cm)

    pair_weights = select_high_confusion_pairs(
        sensitivity,
        K=5
    )

    print("\nHigh Confusion Pairs:")
    print(pair_weights)

    score = confusion_weighted_accuracy(
        y_true,
        y_pred,
        pair_weights
    )

    print("\nConfusion Weighted Accuracy:")
    print(score)


    x_vec = params_to_vector(params)

    history_X.append(x_vec)
    history_y.append(score)

    persistent_pairs = list(pair_weights.keys())


    if len(history_X) >= 3:

        X = np.array(history_X)
        y = np.array(history_y)

        kernel = Matern(nu=2.5)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True
        )

        gp.fit(X, y)

        mu, sigma = gp.predict(
            x_vec.reshape(1, -1),
            return_std=True
        )

        acquisition = confusion_aware_acquisition(
            mu[0],
            sigma[0],
            np.max(y),
            pair_weights,
            persistent_pairs
        )

        print("\nGP Prediction:")
        print("mu =", mu[0])
        print("sigma =", sigma[0])

        print("\nAcquisition Value:")
        print(acquisition)


    if score > best_score:

        best_score = score
        best_params = params

        print("\nNEW BEST!")


print("\n")
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("\nBest Score:")
print(best_score)

print("\nBest Hyperparameters:")
print(best_params)