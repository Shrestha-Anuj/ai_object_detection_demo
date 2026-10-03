# import Streamlit
import streamlit as st

# import the YOLO object-detection model
from ultralytics import YOLO

# import PIL for image processing
from PIL import Image

# import cv2 for image processing
import cv2

# configuring the streamlit page
st.set_page_config(
    page_title="My Object Detection App",
    page_icon=":guardsman:",
    layout="wide"
)

# Application title
st.title("Object Detection Yolov11n")
st.write("Upload an image to detect objects")

# Load the YOLOv11n model
model = YOLO("yolo11n.pt")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# display the uploaded image
if uploaded_file is not None:
   
    # preprocessing the uploaded image
    image = Image.open(uploaded_file)
    
    # run the uploaded image through the YOLOv11n model
    results = model(image)

    # get the results for the uploaded image
    result = results[0]

    # generate annotated image with bounding boxes, confidence scores, and class labels
    annotated_image = result.plot()

    # overwrit the default color code of ultralytics
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # displaying the uploaded image and the detection results side by side
    col1, col2 = st.columns(2)

    # # display the uploaded image
    # col1.image(uploaded_file, caption='Uploaded Image', width=400)

    # # display the results
    # col2.image(annotated_image, caption='Detection Result', width=400)

    # Uploaded image
    col1.markdown(
        "<h3 style='text-align:center;'>Uploaded Image</h3>",
        unsafe_allow_html=True
    )
    col1.image(uploaded_file, width=400)

    # Detection result
    col2.markdown(
        "<h3 style='text-align:center;'>Detection Result</h3>",
        unsafe_allow_html=True
    )
    col2.image(annotated_image, width=400)
