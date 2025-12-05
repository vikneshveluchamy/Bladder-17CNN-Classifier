import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr
import json


# --------------------------------------------------
# Load class names
# --------------------------------------------------
with open("classes.json", "r") as f:
    classes = json.load(f)     # ['HGC','LGC','NST','NTL']


# --------------------------------------------------
# 17-Layer CNN (same structure as Colab training)
# --------------------------------------------------
class CNN17(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
device = "cpu"
model = CNN17(num_classes=len(classes)).to(device)

state = torch.load("final_bladder17.pth", map_location=device)
model.load_state_dict(state, strict=False)
model.eval()


# --------------------------------------------------
# Image Transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict(image):
    image = Image.fromarray(image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Return class + probability dictionary
    result = {classes[i]: float(probs[i]) for i in range(len(classes))}
    predicted_class = classes[torch.argmax(probs).item()]

    return predicted_class, result


# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Label(label="Predicted Class"),
        gr.Label(label="Probabilities")
    ],
    title="Bladder Tissue Classification - 17 Layer CNN",
    description="Upload an endoscopic bladder image."
)

interface.launch()
