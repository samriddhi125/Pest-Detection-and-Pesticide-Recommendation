import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib

main_folder_path = '/Users/samriddhikumari/Desktop/PYthon/Projects/FIPLPROJ/OpenCV-Pest-Detection-master/Datasets/final'

# Initialising lists to store image data, labels, and contour areas
image_data = []
labels = []
list_c_a=[]

# Defining the labels dictionary with class names and their corresponding classes
labels_dict = {'Termite odontotermes (Rambur)':1,'Tetradacus c Bactrocera minax':2}

# Loop through the folders in the main folder
for pest_class in os.listdir(main_folder_path):
    class_path = os.path.join(main_folder_path, pest_class)

    # Check if the path is a directory
    if os.path.isdir(class_path):
        # Loop through the images in each class folder
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            

            # Read the image and convert it to a numpy array, with error handling
            image = cv2.imread(image_path)

            if image is not None:
                # Convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply thresholding to create a binary mask for the pest
                _, binary_mask = cv2.threshold(gray, 120, 190, cv2.THRESH_BINARY_INV)
                
                # Perform morphological operations to clean the binary mask
                kernel = np.ones((3, 3), np.uint8)
                sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)
                
                # Find the sure foreground area
                dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
                
                # Subtract the sure foreground from the sure background
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                
                # Label markers for watershed
                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                
                # Apply watershed algorithm
                markers = cv2.watershed(image, markers)
                image[markers == -1] = [0, 0, 255]  # Highlight watershed boundaries
                
                # Find contours based on watershed result
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                contour_areas = [cv2.contourArea(contour) for contour in contours]
                avg_contour_area = np.mean(contour_areas) if contour_areas else 0

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Adjust the area threshold as needed
                        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)  # Draw in red

                # Append the image data, label, and contour area to their respective lists
                image_data.append(image.flatten())  # Convert the image to a 1D array
                labels.append(pest_class)
                list_c_a.append(avg_contour_area)
                print("Length of image_data:", len(image_data))
                print("Length of labels:", len(labels))
                print("Length of contour_areas:", len(list_c_a))


# Map the class names to their corresponding IDs using the labels_dict
label_ids = [labels_dict[pest_class] for pest_class in labels]

# Create a DataFrame with image data, class labels, and contour areas
data = pd.DataFrame({
    'image_path': image_data,
    'class': labels,
    'label_id': label_ids,
    'contour_area': list_c_a  # Add the contour area as a feature
})

label_1_train = data['label_id'][:290]
features_1_train = data[['contour_area']][:290]

label_2_train = data['label_id'][362:548]
features_2_train = data[['contour_area']][362:548]

label_train = pd.concat([label_1_train, label_2_train], ignore_index = True)
features_train = pd.concat([features_1_train, features_2_train], ignore_index = True)

label_1_test = data['label_id'][290:362]
features_1_test = data[['contour_area']][290:362]

label_2_test = data['label_id'][548:594]
features_2_test = data[['contour_area']][548:594]

label_test = pd.concat([label_1_test, label_2_test], ignore_index = True)
features_test = pd.concat([features_1_test, features_2_test], ignore_index = True)


X_train = features_train
X_test = features_test

Y_train = label_train
Y_test = label_test

model = KNeighborsClassifier(n_neighbors = 3)
model = model.fit(X_train,Y_train)

Y_test_pred = model.predict(X_test) #predicted output
print(f"Validation score is {accuracy_score(Y_test,Y_test_pred) * 100}%") #comparing predicted and actual output

joblib.dump(model, 'knn_model_2.joblib')
