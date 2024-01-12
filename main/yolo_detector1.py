from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO("yolov8n.pt")

# Function to draw bounding boxes on the image frame
def draw_boxes(frame, boxes, name_text):
    annotator = Annotator(frame)
    
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        
        if class_name.lower() == "cat":  # Detect only "cat"
            coordinator = box.xyxy[0]
            annotator.box_label(
                box=coordinator, label=class_name, color=colors(int(class_id), True)
            )

    # Add name to the top-right corner
    font = ImageFont.load_default()
    image_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(image_pil)
    name_position = (image_pil.width - 250, 10)
    draw.text(name_position, name_text, (255, 255, 255), font=font)
    frame = np.array(image_pil)

    return annotator.result()

# Function to detect objects in the image frame
def detect_object(frame, name_text):
    results = model.predict(frame)
    
    for result in results:
        frame = draw_boxes(frame, result.boxes, name_text)

    return frame

if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv.VideoCapture(video_path)

    # Get frame size from the first frame
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame from the video.")
    
    height, width, _ = first_frame.shape

    # Define the codec and create a VideoWriter object with the same size as the original
    video_writer = cv.VideoWriter(
        video_path + "_demo.avi", cv.VideoWriter_fourcc(*"MJPG"), 30, (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Detect objects from the image frame and add name text
            frame_result = detect_object(frame, "Thanatip-Clicknext-Internship-2024")

            # Write the result to the video
            video_writer.write(frame_result)

            # Display the result using matplotlib
            plt.imshow(frame_result[:, :, ::-1])  # Convert BGR to RGB
            plt.show(block=False)
            plt.pause(0.03)

        else:
            break

    # Release the VideoWriter object and close the window
    video_writer.release()
    cap.release()
    cv.destroyAllWindows()
