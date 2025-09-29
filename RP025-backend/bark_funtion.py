import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = ['Healthy', 'HumanCut', 'RoughBark', 'StripeCanker']

class SimCLRModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.base_model = base_model
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])  
        
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),  
            nn.ReLU(),
            nn.Linear(512, projection_dim)  
        )

        self.classification_head = nn.Linear(512, 4)

    def forward(self, x, mode='contrastive'):
        features = self.base_model(x)  
        features = features.view(features.size(0), -1)  
        
        if mode == 'contrastive':
            projection = self.projection_head(features)  
            return projection
        
        elif mode == 'classification':
            return self.classification_head(features)
        
resnet = models.resnet18(pretrained=True)
model = SimCLRModel(resnet)

model.load_state_dict(torch.load('simclr_model_with_humancut_final.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

def predict_image_bark(image_path):
    if not os.path.exists(image_path):
        return None, "File not found"
    
    try:
        img = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        return None, "Invalid image format"
    except Exception as e:
        return None, f"Error opening image: {str(e)}"
    
    img = transform(img).unsqueeze(0).to(device)
    
    try:
        with torch.no_grad():
            output = model(img, mode='classification')
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
            predicted_index = probabilities.argmax()
        
        predicted_label = class_names[predicted_index]
        confidence_scores = max(probabilities)
        
        return predicted_label, confidence_scores
    
    except Exception as e:
        return None, f"Prediction error: {str(e)}"
    

    
# image_path = "E:\\MR.Mind_Projects\\RP025-02\\dataset\\HumanCut\\IMG_H028.jpeg"
# predicted_label, confidence_scores = predict_image_bark(image_path)
# print(f"Predicted Label: {predicted_label}")
# print("Confidence Scores: ", confidence_scores)

image_dir = "E:\\MR.Mind_Projects\\RP025-02\\dataset\\StripeCanker"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for img_name in image_files:
    image_path = os.path.join(image_dir, img_name)
    # print(image_path)
    prediction = predict_image_bark(image_path)
    # if prediction == "HumanCut":
    print(f"Image: {img_name} - Prediction: {prediction}")
        # img = Image.open(image_path)
        # plt.imshow(img)
        # plt.title(f"Predicted: {prediction}")
        # plt.axis('off')
        # plt.show()


