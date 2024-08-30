import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import mediapipe as mp
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


model = HandGestureCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
class_labels = ["up", "down", "left", "right", "select"]


def continuous_learning():
    print("Starting continuous learning mode. Press 'q' to quit.")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )
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
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Manual correction
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    print("Exiting continuous learning mode.")
                    break
                elif key in [ord("u"), ord("d"), ord("l"), ord("r"), ord("s")]:
                    label_index = {"u": 0, "d": 1, "l": 2, "r": 3, "s": 4}[chr(key)]
                    print(f"Corrected to: {class_labels[label_index]}")

                    # Update the model with the corrected label
                    labels = torch.tensor([label_index]).to(device)
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(hand_img)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    torch.save(
                        model.state_dict(), "model_continuous_learning.pth"
                    )  # Save the model after each correction
                    print("Model updated and saved.")

    cap.release()
    cv2.destroyAllWindows()
    print("Continuous learning mode ended.")
