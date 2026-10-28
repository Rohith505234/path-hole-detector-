# 🚀Pothole Detection System

**Project Type:** Deep Learning / Web Application  

## 📌 Project Overview
This is a web-based application that detects potholes in road images using a Deep Learning model (YOLOv4-Tiny). The system allows users to either upload an image or use their live webcam to detect potholes in real-time. It then classifies the severity of the pothole (Low, Medium, High) based on the size of the bounding box. 

## 🛠️ Technologies Used
- **Backend:** Python, Flask
- **Machine Learning:** YOLOv4-Tiny (Darknet), OpenCV
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)

## 🧠 How It Works
1. The user uploads an image or captures a frame from the webcam.
2. The image is sent to the Flask backend (`/detect` route).
3. OpenCV preprocesses the image (resizes it to 640x480) and passes it to the YOLOv4-Tiny model.
4. The model returns bounding boxes and confidence scores.
5. The backend calculates the area of the pothole to assign a severity level:
   - **Low:** Small bounding box area.
   - **Medium:** Medium bounding box area.
   - **High:** Large bounding box area.
6. The result is sent back to the frontend, where a green box is drawn around the pothole with a label showing the confidence score and severity.

## 🛠️ My Contributions & Troubleshooting
- **OpenCV Compatibility Fix:** The project originally failed to run on OpenCV 5.0 because the Darknet importer was removed. I troubleshooted the error, downgraded to OpenCV 4.14, and successfully loaded the model.
- **Backend Optimization:** I renamed the model variable to `pothole_model` for better readability and adjusted the confidence threshold to `0.6` to reduce false positives.
- **Frontend Redesign:** I completely redesigned the user interface using modern CSS (Dark Blue theme) to make it clean and user-friendly. I also added functionality for real-time webcam detection.

## ⚙️ How to Run the Project Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/[YourUsername]/[YourRepoName].git
