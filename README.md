# 🥭 Mango Leaf Disease Classification

## 📁 Cấu trúc Dataset

```
MangoLeaf_Dataset/
├── Anthracnose/          # Bệnh thán thư
├── Bacterial_Canker/     # Bệnh loét vi khuẩn
├── Bacterial_Spot/       # Bệnh đốm vi khuẩn
├── Cutting_Weevil/       # Sâu đục thân (Mọt cắt)
├── Die_Back/             # Bệnh chết ngọn
├── Gall_Midge/           # Sâu đục chồi (Muỗi gây u)
├── Healthy/              # Lá khỏe mạnh
├── Powdery_Mildew/       # Bệnh phấn trắng
└── Sooty_Mould/          # Bệnh muội đen
```

## 🦠 Mô tả các loại bệnh

| Tên tiếng Anh | Tên tiếng Việt | Mô tả |
|---------------|----------------|-------|
| **Anthracnose** | Bệnh thán thư | Gây ra các vết đốm nâu đen trên lá, thường xuất hiện khi thời tiết ẩm ướt. Làm lá khô héo và rụng sớm. |
| **Bacterial_Canker** | Bệnh loét vi khuẩn | Tạo các vết loét, nứt nẻ trên lá và cành. Gây chảy nhựa và làm chết mô cây. |
| **Bacterial_Spot** | Bệnh đốm vi khuẩn | Xuất hiện các đốm nhỏ màu nâu hoặc đen trên bề mặt lá, có quầng vàng xung quanh. |
| **Cutting_Weevil** | Sâu đục thân (Mọt cắt) | Côn trùng gây hại bằng cách cắt và đục vào thân, cành non. Lá bị héo do mất nước. |
| **Die_Back** | Bệnh chết ngọn | Ngọn cành bị khô héo và chết dần từ đầu vào. Thường do nấm hoặc vi khuẩn gây ra. |
| **Gall_Midge** | Sâu đục chồi (Muỗi gây u) | Ấu trùng muỗi xâm nhập vào chồi non, gây biến dạng và tạo u sưng trên lá. |
| **Healthy** | Lá khỏe mạnh | Lá xoài bình thường, không có dấu hiệu bệnh tật hay sâu hại. |
| **Powdery_Mildew** | Bệnh phấn trắng | Lớp bột trắng như phấn phủ trên bề mặt lá. Làm lá biến dạng và giảm quang hợp. |
| **Sooty_Mould** | Bệnh muội đen | Lớp nấm đen phủ trên lá, thường do côn trùng chích hút tiết mật gây ra. Cản trở quang hợp. |

## 📊 Thông tin Dataset

- **Tổng số lớp:** 9 (8 bệnh + 1 khỏe mạnh)
- **Kích thước ảnh:** 240 x 240 pixels
- **Định dạng:** RGB (3 kênh màu)

## 🔧 Kỹ thuật sử dụng

- **Data Pipeline:** tf.data.Dataset (thay thế ImageDataGenerator)
- **Augmentation:** Keras Preprocessing Layers (GPU-accelerated)
- **Training:** Two-stage (Feature Extraction → Fine-tuning)
- **Explainability:** Grad-CAM (Explainable AI)
- **Models:** EfficientNetB0, MobileNetV2, ResNet50, VGG16, DenseNet121, InceptionV3

## 📱 Export

- `mango_leaf_model.keras` - Model đầy đủ
- `mango_leaf_model.tflite` - Model tối ưu cho Mobile
- `labels.txt` - Danh sách nhãn

---
*Dự án nhận diện bệnh trên lá xoài để hỗ trợ nông dân*
