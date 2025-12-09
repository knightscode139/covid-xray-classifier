import torch
import torch.nn as nn
from torchvision import transforms, models
import gradio as gr

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
class_names = ['covid', 'lung-opacity', 'normal', 'viral-pneumonia']

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('covid_xray_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define transforms
val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    img_tensor = val_test_transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Create results dictionary
    results = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return results

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="COVID-19 Chest X-Ray Classifier",
    description="Upload a chest X-ray image to classify: COVID-19, Lung Opacity, Normal, or Viral Pneumonia. Test Accuracy: 92%"
)

if __name__ == "__main__":
    interface.launch()
