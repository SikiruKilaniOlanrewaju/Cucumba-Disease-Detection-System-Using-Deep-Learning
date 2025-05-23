# app.py

"""
Flask web app for cucumber plant disease detection using the trained PyTorch model.
"""

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image

# Import the model definition from main.py (or copy the SimpleNet class here)
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

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load class names (adjust as needed)
# Use the same class order as in your dataset
class_names = ['Ill_cucumber', 'good_Cucumber']

# Load the trained model
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

def predict(image_path):
    image = Image.open(image_path)
    tensor = transform_image(image)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict(filepath)
            return render_template('result.html', prediction=prediction, filename=filename)
    return render_template('index.html')

# In result.html, the image is referenced as /uploads/{{ filename }}
# To serve uploaded files, add a static route
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
