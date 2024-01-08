import streamlit as st
from PIL import Image
import torch
from pathlib import Path
import os
import cv2
from torchvision import transforms
import time

# YOLOv5 modelini yükleme
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5x', pretrained=True)

def object_detection(image):
    results = model(image)

    output_path = "C:\\Users\\burak\\Desktop\\deneme3"
    results.save(output_path)

    return results, output_path





def main():
    st.title("OBject Detection")
    result = st.button("ReadlTime Object Detection")
    if result:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            pred = results.xyxy[0].numpy()
            for det in pred:
                cv2.rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[int(det[5])]}: {det[4]:.2f}', (int(det[0]), int(det[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('Real-Time YoloV5 Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    # Fotoğraf üzerinden nesne tespiti
    st.title("YOLOv5 Object Detection")
    uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Yüklenen resmi gösterme
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        # Nesne tespiti
        results, _ = object_detection(original_image)
        output_path = "C:\\Users\\burak\\Desktop\\deneme3\\image0.jpg"
        output_image = Image.open(output_path)
        
        # İşlenmiş resmi gösterme
        st.image(output_image, caption="Model Image", use_column_width=True)

if __name__ == "__main__":
    main()
