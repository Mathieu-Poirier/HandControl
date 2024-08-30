import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import cv2
import mediapipe as mp
import numpy as np
import torch.nn.functional as F
import time
from tqdm import tqdm  # Progress bar

# Define the CNN model
class HandGestureCNN(nn.Module):
    def __init__(self):
        super(HandGestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        # Adjust the input size of the fully connected layer for 600x600 images
        self.fc1 = nn.Linear(64 * 150 * 150, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 classes for the gestures

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 150 * 150)  # Adjust based on the feature map size
        x = self.dropout(F.relu(self.fc1(x)))  # Apply dropout
        x = self.fc2(x)
        return x

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load dataset and split into train, unseen, and validation sets
print("Loading and splitting dataset...")
start_time = time.time()
dataset = datasets.ImageFolder(root='dataset', transform=transform)
train_size = int(0.6 * len(dataset))
unseen_size = int(0.2 * len(dataset))
val_size = len(dataset) - train_size - unseen_size
train_dataset, unseen_dataset, val_dataset = random_split(dataset, [train_size, unseen_size, val_size])
print(f"Dataset loaded and split in {time.time() - start_time:.2f} seconds.")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
unseen_loader = DataLoader(unseen_dataset, batch_size=1, shuffle=False)  # Batch size 1 for manual correction
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the model, loss function, and optimizer
model = HandGestureCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Class labels
class_labels = ['up', 'down', 'left', 'right', 'select']
label_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'select'}

# Class labels
class_labels = ['up', 'down', 'left', 'right', 'select']

# Training loop with tqdm progress bar
print("Starting training phase...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
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
    
    print(f'Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f} seconds')

torch.save(model.state_dict(), 'model_after_training.pth')


# Function to manually correct predictions
def correct_predictions(unseen_loader):
    print("Starting correction phase on unseen dataset...")
    model.eval()
    corrected_labels = []
    total_images = len(unseen_loader)
    for i, (images, _) in enumerate(unseen_loader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Display the image and prediction
        hand_img_np = images.cpu().numpy().squeeze().transpose(1, 2, 0)
        hand_img_np = (hand_img_np * 255).astype(np.uint8)
        hand_img_np = cv2.cvtColor(hand_img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
        cv2.imshow('Image', hand_img_np)
        print(f'Predicted: {class_labels[predicted.item()]}')
        print(f"Label Mapping: {label_map}")
        print(f"Images left to label: {total_images - i - 1}")
        
        while True:
            try:
                correct_label = int(input("Enter the correct label index (or -1 to skip): "))
                if correct_label == -1 or (0 <= correct_label < len(class_labels)):
                    break
                else:
                    print(f"Invalid input. Please enter a number between 0 and {len(class_labels) - 1}, or -1 to skip.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        if correct_label == -1:
            corrected_labels.append(predicted.item())  # If skipped, keep the predicted label
        else:
            corrected_labels.append(correct_label)
        cv2.waitKey(3)  # Add a small wait time for OpenCV to render the image
    
    cv2.destroyAllWindows()
    print("Correction phase completed.")
    return corrected_labels

# Correct the unseen set and save the model
corrected_labels = correct_predictions(unseen_loader)
torch.save(model.state_dict(), 'model_after_correction.pth')


# Update the unseen_dataset with corrected labels
print("Updating unseen dataset with corrected labels...")
for i, (img, _) in enumerate(unseen_dataset):
    unseen_dataset[i] = (img, corrected_labels[i])

# Optionally fine-tune the model with the updated dataset
print("Starting fine-tuning phase...")
fine_tune_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=True)
fine_tune_epochs = 5
for epoch in range(fine_tune_epochs):
    model.train()
    fine_tune_start_time = time.time()
    for images, labels in fine_tune_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Fine-tuning Epoch [{epoch+1}/{fine_tune_epochs}], Time: {time.time() - fine_tune_start_time:.2f} seconds')

print("Fine-tuning phase completed.")

# Validate the model
print("Starting validation phase...")
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
print(f'Validation Accuracy: {val_accuracy:.4f}')
print("Validation phase completed.")

# Continuous learning phase using webcam
def continuous_learning():
    print("Starting continuous learning mode. Press 'q' to quit.")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                h, w, _ = frame.shape
                x_min = int(x_min * w) - 30
                x_max = int(x_max * w) + 30
                y_min = int(y_min * h) - 30
                y_max = int(y_max * h) + 30

                x_min = max(0, x_min)
                x_max = min(w, x_max)
                y_min = max(0, y_min)
                y_max = min(h, y_max)

                hand_img = frame[y_min:y_max, x_min:x_max]
                hand_img = cv2.resize(hand_img, (600, 600))
                hand_img = hand_img.astype(np.float32) / 255.0
                hand_img = np.transpose(hand_img, (2, 0, 1))  # Convert to CHW format
                hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension
                hand_img = torch.tensor(hand_img).to(device)

                outputs = model(hand_img)
                _, predicted = torch.max(outputs, 1)
                gesture = class_labels[predicted.item()]
                print(f"Predicted gesture: {gesture}")
                cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Manual correction
                cv2.imshow('Frame', frame)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("Exiting continuous learning mode.")
                    break
                elif key in [ord('u'), ord('d'), ord('l'), ord('r'), ord('s')]:
                    label_index = {'u': 0, 'd': 1, 'l': 2, 'r': 3, 's': 4}[chr(key)]
                    print(f'Corrected to: {class_labels[label_index]}')

                    # Update the model with the corrected label
                    labels = torch.tensor([label_index]).to(device)
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(hand_img)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    torch.save(model.state_dict(), 'model_continuous_learning.pth')  # Save the model after each correction
                    print("Model updated and saved.")

    cap.release()
    cv2.destroyAllWindows()
    print("Continuous learning mode ended.")

# Start continuous learning phase
continuous_learning()

