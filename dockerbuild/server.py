from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
from collections import Counter
from standard_model import HandGestureCNN


app = Flask(__name__)

model = HandGestureCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    model.load_state_dict(
        torch.load(
            "converged_CNN.pth", map_location=torch.device("cuda"), weights_only=True
        )
    )
else:
    model.load_state_dict(
        torch.load(
            "converged_CNN.pth", map_location=torch.device("cpu"), weights_only=True
        )
    )
print("OK: Loaded model")

# Move the model to the correct device
model.to(device)
model.eval()

# Define the labels
class_labels = ["up", "down", "left", "right", "select"]

predictions = []


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"].read()

        image = Image.open(io.BytesIO(file)).convert("RGB")
        image = image.resize((600, 600), Image.ANTIALIAS)

        image = np.array(image).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = torch.tensor(image, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            gesture = class_labels[predicted.item()]

        # Update predictions list
        predictions.append(gesture)
        if len(predictions) > 5:
            predictions.pop(0)

        # Determine the most frequent prediction
        most_common_prediction = Counter(predictions).most_common(1)[0][0]

        return jsonify({"gesture": most_common_prediction})

    except Exception as e:
        print("An error occurred:", e)
        return "Internal Server Error", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
