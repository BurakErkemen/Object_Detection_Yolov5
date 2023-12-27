import streamlit as st
from PIL import Image
import torch
from pathlib import Path
import os
import cv2
from torchvision import transforms

# YOLOv5 modelini yükleme
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5x', pretrained=True)

def object_detection(image, output_directory):
    # Modeli kullanarak resmi işleme
    results = model(image)

    # İşlenmiş resmi kaydetme
    output_filename = f"output_{os.path.splitext(os.path.basename(image.filename))[0]}.jpg"
    output_path = output_filename
    results.save(output_path)

    return results, output_path

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
def real_time_object_detection():

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Kameradan bir frame al
        ret, frame = cap.read()

        # YOLOv5 için frame'i dönüştür
        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_image = transform(input_image)
        input_image = input_image.unsqueeze(0)

        # Nesne tespiti
        results = model(input_image)

        # Tespit edilen nesnelerin çizimleri
        for det in results[0]:
            bbox = det[0:4].cpu().numpy().astype(int) 
            class_id = int(det[5].item())
            confidence = det[4].item()

        # Eğer güven düzeyi belirli bir eşik değerinden büyükse
            if confidence > 0.5:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Class: {class_id}, Confidence: {confidence:.2f}", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Streamlit üzerinde görüntüleme
        st.image(frame, channels="BGR", caption="Real-time Object Detection", use_column_width=True)

        # Streamlit uygulamasının güncellenmesini bekleyin
        st.empty()

        # Geri dönmek istenirse
        if st.button("Fotoğraf Üzerinden Nesne Tanıma"):
            cap.release()
            cv2.destroyAllWindows()
            return

def main():
    # Sayfanın üst kısmına buton eklemek için HTML ve CSS kullanımı
    st.markdown(
    """
    <style>
        .stButton button {
            padding: 10px;
            font-size: 16px;
            color: white;
            background-color: #4CAF50;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton button span {
            display: inline-block;
            visibility: visible;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

    # Gerçek Zamanlı Nesne Tanıma butonu
    if st.button("Gerçek Zamanlı"):
        st.write("Gerçek Zamanlı Nesne Tanımlama Başlatılıyor")
        real_time_object_detection()

    # Fotoğraf üzerinden nesne tespiti
    st.title("YOLOv5 Object Detection1 ")
    uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Yüklenen resmi gösterme
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        # Nesne tespiti
        results, _ = object_detection(original_image, "")
        
        # İşlenmiş resmi gösterme
        st.image(results.show().to_pil(), caption="Model Image", use_column_width=True)

if __name__ == "__main__":
    main()
