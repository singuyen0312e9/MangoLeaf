

import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from torchvision import models, transforms
from PIL import Image
import io
import cv2
import numpy as np

app = Flask(__name__)




# --- 1. ĐỊNH NGHĨA KIẾN TRÚC MOBILENETV2 CẢI TIẾN ---
class EnhancedMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedMobileNetV2, self).__init__()
        # Load backbone không weights để load từ file pth
        self.backbone = models.mobilenet_v2(weights=None).features
        
        # Squeeze-and-Excitation Attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, 80, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(80, 1280, kernel_size=1),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x * self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def load_enhanced_mbn(num_classes, model_path, device):
    model = EnhancedMobileNetV2(num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Success] Loaded Enhanced MobileNetV2 from {model_path}")
    else:
        print(f"[Error] Model not found at {model_path}")
    model = model.to(device)
    model.eval()
    return model

# --- 2. CẤU HÌNH MODEL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 9
MODEL_PATH = "../Result/MBN_Caitien/best_model.pth"

model = load_enhanced_mbn(num_classes, MODEL_PATH, DEVICE)



LABELS = [
    "Thán thư (Anthracnose)", 
    "Loét vi khuẩn (Bacterial Canker)", 
    "Đốm đen (Bacterial Spot)", 
    "Sâu ăn lá (Cutting Weevil)", 
    "Chết cành (Die Back)", 
    "Trĩ (Gall Midge)", 
    "Khỏe mạnh (Healthy)", 
    "Phấn trắng (Powdery Mildew)", 
    "Bồ hóng (Sooty Mould)"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def leaf_focus_crop(img_pil):
    """
    Tự động tìm vùng có lá (màu xanh) và crop hình vuông quanh đó.
    Dựa trên logic từ file rebuild_dataset_dedup.py.
    """
    # Chuyển từ PIL sang OpenCV
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h_img, w_img = img.shape[:2]
    
    # Chuyển sang không gian màu HSV để lọc màu xanh lá
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    
    # Tìm các đường bao (contours) của vùng màu xanh
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    s = min(h_img, w_img)
    
    if contours:
        # Lấy contour có diện tích lớn nhất (lá rõ nhất)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h_rect = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h_rect // 2
        
        # Tính toán tọa độ crop hình vuông s x s quanh tâm cx, cy
        x1 = max(0, min(cx - s // 2, w_img - s))
        y1 = max(0, min(cy - s // 2, h_img - s))
        cropped = img[y1:y1+s, x1:x1+s]
    else:
        # Nếu không tìm thấy màu xanh, thực hiện center crop như cũ
        x1, y1 = (w_img - s) // 2, (h_img - s) // 2
        cropped = img[y1:y1+s, x1:x1+s]
    
    # Chuyển ngược lại PIL RGB
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    img = leaf_focus_crop(img)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    
    return jsonify({
        'prediction': LABELS[idx.item()],
        'confidence': f"{conf.item()*100:.2f}%",
        'all_probs': {LABELS[i]: f"{probs[i].item()*100:.1f}%" for i in range(len(LABELS))}
    })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    print(f"Server is running with ENHANCED MOBILENETV2 on {DEVICE}...")
    app.run(debug=True, port=5000)

