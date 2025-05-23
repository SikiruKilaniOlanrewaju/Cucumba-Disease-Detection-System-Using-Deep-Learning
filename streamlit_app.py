import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# Model definition (same as in main.py)
class SimpleNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(128 * 128 * 3, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Class names (order must match training)
class_names = ['Ill_cucumber', 'good_Cucumber']

# Load model
num_classes = len(class_names)
model = SimpleNet(num_classes)
model_path = os.path.join(os.getcwd(), 'cucumber_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def predict(image):
    tensor = transform_image(image)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Streamlit UI
st.title('Cucumber Disease Detection')
st.write('Upload a cucumber leaf image to predict its health status.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Predicting...')
    prediction = predict(image)
    st.success(f'Prediction: {prediction}')
