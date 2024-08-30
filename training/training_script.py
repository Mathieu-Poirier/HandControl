import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
from tqdm import tqdm  # Progress bar
from learning_stages import continuous_learning
from standard_model import HandGestureCNN


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"OK: Using device: {device}")

transform = transforms.Compose(
    [
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalization
    ]
)

start_time = time.time()
dataset = datasets.ImageFolder(root="dataset", transform=transform)
train_size = int(0.6 * len(dataset))
unseen_size = int(0.2 * len(dataset))
val_size = len(dataset) - train_size - unseen_size
train_dataset, unseen_dataset, val_dataset = random_split(
    dataset, [train_size, unseen_size, val_size]
)
print(f"OK: Dataset loaded and split in {time.time() - start_time:.2f} seconds.")


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
unseen_loader = DataLoader(
    unseen_dataset, batch_size=1, shuffle=False
)  # Batch size 1 for manual correction
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("OK: Created loaders")

model = HandGestureCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("OK: Set optimzer and set model device")

class_labels = ["up", "down", "left", "right", "select"]
label_map = {0: "up", 1: "down", 2: "left", 3: "right", 4: "select"}


print("OK: Starting training phase...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
    with tqdm(
        total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
    ) as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
            pbar.update(1)

    print(
        f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f} seconds"
    )

torch.save(model.state_dict(), "model_after_training.pth")

# Validate the model
print("OK: Starting validation phase...")
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = correct / total
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("OK: Validation phase completed.")


continuous_learning()
