import streamlit as st
from PIL import Image
import torch
from pathlib import Path
import os
import cv2
from torchvision import transforms

# YOLOv5 modelini yükleme
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5x', pretrained=True)

def object_detection(image):
    # Modeli kullanarak resmi işleme
    results = model(image)

    # İşlenmiş resmi kaydetme
    output_path = "C:\\Users\\burak\\Desktop\\deneme3"
    results.save(output_path)

    return results, output_path


def RealTime_Object_Detection(frame):
    
    return frame



def main():
    st.title("Realtime OBject Detection")

    



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
