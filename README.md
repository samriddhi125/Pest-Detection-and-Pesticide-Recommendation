# Pest Detection and Pesticide Recommendation System

## Introduction
Our project aims to enhance farming techniques for farmers by utilizing image analysis to detect pests on crops and reduce the use of pesticides. Pest detection and counting is typically a labour intensive and time-consuming task for farmers. By automating the process through image analysis, farmers can monitor the crops more efficiently, apply targeted pest management strategies, limit the usage of pesticides, and improve both crop quantity and quality.

## Meet the Team
We are a team of inquisitive undergraduate students pursuing B.Tech in AIML from Symbiosis Institute of Technology - Pune, India.

- Rampalli Agni Mithra 
- Samriddhi Kumari
- V Yasaswini

## Problem Statement
The agricultural sector grapples with inefficiencies in pest detection and management, impacting crop health and overall productivity. Our project aims to address these challenges by leveraging image processing techniques to automate and enhance pest detection, offering a solution for more effective crop management and ultimately elevating agricultural productivity.

## Dataset Decription
Dataset consists of 56,685 high-quality images sorted into 132 pest classes, along with relevant information on the pesticides used to control them.
*Dataset Source:*
https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/grayscale

## Plan of Action
**Data Reading, Preprocessing and Labeling:**
Using os.listdir we have read each file from its labeled folder and labeled it with the appropriate class and label_id for later classification. All images read were then flattened into 1-D arrays, and subsequently added to a dataframe.

After reading the image from the file directory we have performed preprocessing step:
- Grayscale conversion
- Thresholding to create a binary mask
- Morphological operations to clean the binary mask
- Watershed segmentation algorithm
- Average contour area calculation
After preprocessing, we calculated the average contour area for each image, which was considered as the primary feature for classification, which was also added to the DataFrame.

**K-Nearest Neighbors (KNN) Classification:**

We used K-Nearest Neighbour classifier with n_neighbors = 3, from the scikit-learn library as the model to be trained. We split the data into training (80%) and testing (20%) sets. Then the model was trained on the data provided, and using joblib the trained model was saved to be deployed by Streamlit.

**Model Evaluation:**

The accuracy score was calculated by comparing the predicted labels with the actual labels in the test data. The validation score indicates the accuracy of the model's predictions. We achieved an Accuracy score of 65.035%.

### Technologies Used:
Python libraries such as 
- sciKit-image
- SimpleITK
- Matplotlib
- OpenCV
- Numpy
- Pandas
- StreamLit

### Results: 
Our pest detection system was able to achieve an accuracy of 65.035% on the test data. This accuracy explains the pest detection ability of the model.

### Limitations and Future Work
- Feature engineering can be performed more accurately with additional characteristics and metadata for better pest detection.
- Future analysis and model building can be done using various other deep learning techniques and complex modeling techniques.

## Get Involved
We encourage you to get in touch if our project interests you and you're eager to help, offer comments, or look into potential collaborations. We appreciate your input and encouragement very much, and we look forward to working with you to further our project.

## Acknowledgments
We'd like to express our gratitude to our mentor Dr.Jayant Jagtap for his guidance and support throughout the project.

Thank you for taking the time to explore our mini-project. We're thrilled about the prospect of creating a positive impact.
