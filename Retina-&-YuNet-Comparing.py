import cv2
import time
import numpy as np
import os
from retinaface import RetinaFace
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load YuNet Model from ONNX
yunet_model_path = r"face_detection_yunet_2023mar.onnx"
yunet = cv2.FaceDetectorYN.create(yunet_model_path, "CUDA", (320, 320), 0.6, 0.3, 12)

# Folder containing the dataset of images
dataset_path = r"Dataset"
image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Input the total expected number of faces for the entire dataset
expected_faces_total = int(input("Enter the expected total number of faces in the dataset: "))

# Initialize performance tracking variables
total_yunet_time, total_retina_time = 0, 0
yunet_total_faces, retina_total_faces = 0, 0

print(f"Processing {len(image_files)} images from the dataset...")

# Loop through the dataset with a progress bar
for image_path in tqdm(image_files, desc="Processing Dataset"):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    yunet.setInputSize((width, height))

    # Run face detection with YuNet
    start = time.time()
    success, faces = yunet.detect(image)  # Ensure successful detection

    yunet_boxes = []
    if success and faces is not None and len(faces) > 0:
        for face in faces:
            x, y, w, h = face[:4]
            yunet_boxes.append([int(x), int(y), int(x + w), int(y + h)])

    yunet_time = (time.time() - start) * 1000  # Convert to milliseconds
    total_yunet_time += yunet_time
    yunet_total_faces += len(yunet_boxes)

    # Run face detection with RetinaFace
    start = time.time()
    rf_faces = RetinaFace.detect_faces(image)
    retina_boxes = [face_info["facial_area"] for face_info in rf_faces.values()] if isinstance(rf_faces, dict) else []
    retina_time = (time.time() - start) * 1000  # Convert to milliseconds
    total_retina_time += retina_time
    retina_total_faces += len(retina_boxes)

    # Crop and visualize detected faces
    for i, (model_name, boxes) in enumerate([("YuNet", yunet_boxes), ("RetinaFace", retina_boxes)]):
        faces_cropped = [image[int(y):int(y2), int(x):int(x2)] for (x, y, x2, y2) in boxes]
        if faces_cropped:
            fig, axes = plt.subplots(1, len(faces_cropped), figsize=(15, 5))
            if len(faces_cropped) == 1:
                axes = [axes]
            for j, face in enumerate(faces_cropped):
                axes[j].imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                axes[j].axis('off')
            plt.suptitle(f"{model_name} - {os.path.basename(image_path)}")
            plt.show()
        else:
            print(f"No faces detected using {model_name} in {os.path.basename(image_path)}")

# Calculate accuracy
yunet_accuracy = (yunet_total_faces / expected_faces_total) * 100 if expected_faces_total > 0 else 0
retina_accuracy = (retina_total_faces / expected_faces_total) * 100 if expected_faces_total > 0 else 0

# Print total time and accuracy results
print("\n=== Summary ===")
print(f"Total images processed: {len(image_files)}")
print(f"Expected total faces: {expected_faces_total}")
print(f"YuNet: {total_yunet_time:.2f} ms total, {yunet_total_faces} faces detected, Accuracy: {yunet_accuracy:.2f}%")
print(f"RetinaFace: {total_retina_time:.2f} ms total, {retina_total_faces} faces detected, Accuracy: {retina_accuracy:.2f}%")
