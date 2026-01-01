
import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# --- 1. ĐỊNH NGHĨA KIẾN TRÚC SHUFFLENETV2 + CBAM (ATTENTION) ---
# Module này phải khớp hoàn toàn với kiến trúc đã train trong Bước 8 & 9
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))

class CBAMShuffleNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.shufflenet_v2_x1_0(weights=None)
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5
        self.ca = ChannelAttention(464) 
        self.sa = SpatialAttention()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x); x = self.maxpool(x)
        x = self.stage2(x); x = self.stage3(x)
        x = self.stage4(x)
        x = x * self.ca(x)
        x = x * self.sa(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

# --- 2. CẤU HÌNH MODEL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 9
model = CBAMShuffleNetV2(num_classes).to(DEVICE)

# Path tới file model Attention mới nhất
MODEL_PATH = "../Result/ShuffleNetV2_CBAM_Improved/best_shufflenet_cbam.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"[Success] Loaded ShuffleNetV2 + CBAM Attention from {MODEL_PATH}")
else:
    print(f"[Error] Model not found at {MODEL_PATH}")

model.eval()

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

def center_crop_pillow(img):
    w, h = img.size
    s = min(w, h)
    left = (w - s) / 2
    top = (h - s) / 2
    return img.crop((left, top, left + s, top + s))

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
    
    img = center_crop_pillow(img)
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
    print(f"Server is running with ATTENTION MODEL on {DEVICE}...")
    app.run(debug=True, port=5000)
