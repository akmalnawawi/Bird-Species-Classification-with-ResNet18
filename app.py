import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Initialize and load the model
model = models.resnet18(pretrained=False)  # Set pretrained=False to load your own weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Number of classes
model.load_state_dict(torch.load('birds_resnet18.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a function to make predictions
def predict(image, model):
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Streamlit interface
st.title("Bird Species Classifier")
st.write("Upload an image of a bird, and the model will predict its species.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict the class of the image
    prediction = predict(image, model)
    
    # Display the prediction
    bird_species = {0: 'Species A', 1: 'Species B', 2: 'Species C', 3: 'Species D'}
    st.write(f"Prediction: {bird_species[prediction]}")
