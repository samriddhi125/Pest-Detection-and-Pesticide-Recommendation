import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import cv2
from PIL import Image
#streamlit run /Users/samriddhikumari/Desktop/PYthon/Projects/FIPLPROJ/stream1.py

model = joblib.load('knn_model_2.joblib')

def main():
    #
    st.title('PEST DETECTION AND PESTICIDE RECOMMENDATION')
    
    st.sidebar.header("Upload Pest Image")
    uploaded_image = st.sidebar.file_uploader("Upload")
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        image = np.array(image)
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary mask for the pest
        _, binary_mask = cv2.threshold(gray, 120, 190, cv2.THRESH_BINARY_INV)

        # Convert the binary mask to the correct format (CV_8UC1)
        binary_mask = binary_mask.astype(np.uint8)

        # Find contours based on watershed result
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        avg_contour_area = np.mean(contour_areas) if contour_areas else 0
        prediction = model.predict(np.array([[avg_contour_area]]))

        if prediction == 1:
            st.write("Termite odontotermes (Rambur)")
            st.write("Pesticide Recommeded: Chlorpyrifos, Imidacloprid, Fiproni")
        else:
            st.write("Tetradacus c Bactrocera minax")
            st.write("Pesticide Recommeded: Malathion, Spinosad, Imidacloprid")

if __name__ == '__main__':
    main()
