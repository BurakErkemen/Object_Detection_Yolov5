import streamlit as st
import cv2
from PIL import Image
from pathlib import Path
from io import BytesIO
import numpy as np
from torchvision import transforms
import torch

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)

# Giriş resminin boyutunu ve dönüşümünü belirle
img_size = 640
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

def detect_objects(image):
    # OpenCV karesini RGB renk uzayına dönüştür
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resmi numpy dizisinden Torch Tensor'a dönüştür
    input_image = torch.from_numpy(np.array(rgb_image)).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0

    # Nesne tespiti
    results = model(input_image)

    # Tespit edilen nesnelerin çizimleri
    for det in results[0]:
        bbox = det[0:4].cpu().numpy().astype(int)
        class_id = int(det[5].item())
        confidence = det[4].item()

        if confidence > 0.5:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, f"Class: {class_id}, Confidence: {confidence:.2f}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def main():
    st.title("Streamlit Kamera ve Nesne Tanıma Örneği")

    # Kamera açma için buton
    if st.button("Kamerayı Aç"):
        # Kamera akışını başlat
        cap = cv2.VideoCapture(0)
        
        # Kamera açıldıysa devam et
        if cap.isOpened():
            st.success("Kamera başarıyla açıldı!")
        else:
            st.error("Kamera açılamadı. Lütfen bağlantıları kontrol edin.")
            return

        # Streamlit'te görüntülemek için st.image kullanılır
        image_placeholder = st.empty()

        # Streamlit uygulamasının güncellenmesini bekleyin
        st.text("Kamera akışını durdurmak için 'Kamerayı Kapat' butonuna tıklayın.")
        while st.button("Kamerayı Kapat") is False:
            # Kameradan bir frame alın
            ret, frame = cap.read()

            if not ret:
                st.warning("Kamera akışından frame alınamadı. Çıkış yapılıyor...")
                break

            # Görüntü üzerinde nesne tanıma
            frame_with_objects = detect_objects(frame)

            # Görüntüyü Streamlit uygulamasında göster
            image_placeholder.image(frame_with_objects, channels="BGR", use_column_width=True)

        # Kamera bağlantısını serbest bırak
        cap.release()

if __name__ == "__main__":
    main()
