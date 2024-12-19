import logging
import sqlite3
import time
import os
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from constants import (NUMBER_PLATE_DETECTION_SCORE, CLIPS_PATH, CSV_FILE_PATH, MODEL_PATH, YOLO_MODEL, PROCESSING_TABLE_SCHEMA)
from constants import *
import onnxruntime as ort
import csv
import jellyfish
from paddleocr import PaddleOCR
import cv2  
from ultralytics import YOLO 




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
event_id = False
event_data = []
number_plate_processing_ids = []
thread_inprocess = False
number_plate_processed_ids =[]
camera_name = "None"




ocr = PaddleOCR(use_angle_cls=True, lang='en')  # PaddleOCR with angle classification for English



# Define the CSV file path
csv_file_path = CSV_FILE_PATH
camera_name = "None"

# Initialize a list to hold the data
car_label_list = []
car_bbox = []

with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Read the first row (car names)
    car_names = next(csv_reader)

    # Read the second row (license plates) directly as a list
    license_plates = next(csv_reader)

    # Trim any excess whitespace from the car names and license plates
    car_names = [name.strip() for name in car_names]
    license_plates = [plate.strip() for plate in license_plates]

    # Debugging: Print the lengths and contents
    print(f"[Number-plate-detection] - Car names: {car_names}, Length: {len(car_names)}")
    print(f"[Number-plate-detection] - License plates: {license_plates}, Length: {len(license_plates)}")

    # Ensure that both lists have the same length before proceeding
    if len(car_names) != len(license_plates):
        raise ValueError("Mismatch between the number of car names and license plates in the CSV file.")

    # Create a mapping of car names to license plates
    car_label_list = [{car_names[i]: license_plates[i]} for i in range(len(car_names))]

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH)
# Class names: "parcel" and "number plate"
class_names = ['parcel', 'number plate', 'car']
conf_threshold = NUMBER_PLATE_DETECTION_SCORE


def generate_number_plate_bbox (event_id, event_data, cam_name):
    global camera_name, car_bbox
    camera_name = cam_name
    file_path_main_image = os.path.join(CLIPS_PATH, f"{camera_name}-{event_id}-number-plate.png")
    try:

        read_attempts = 5
        interval_seconds = 1

        for attempt in range(read_attempts):
            start_time = time.time()
            img = cv2.imread(file_path_main_image)
            
            if img is None:
                print(f"Attempt {attempt + 1}: Image not found.")
            else:
                print(f"Attempt {attempt + 1}: Image read successfully.")
                break
            # Use a time-based loop to wait for 5 seconds without using sleep
            while time.time() - start_time < interval_seconds:
                pass

        if img is None:
            print("[Number-plate-detection] - Image not loaded. Check the path")
            return False, None
        else:
            print("[Number-plate-detection] - Recevied call for number plate recognition: ", event_id)
            car_bbox = event_data['snapshot']['box']  
            print(car_bbox)
            file_path, best_match_name, ocr_text = main_processing(img, event_id,event_data)
            print("[Number-plate-detection] - File path found: ", file_path)
            print("[Number-plate-detection] - Best match name found: ", best_match_name)
            print("[Number-plate-detection] - Ocr text found: ", ocr_text)
            #car_bbox = event_data['snapshot']['box']
            return file_path, best_match_name, ocr_text
    except:
        print("[Number-plate-detection] - Error in image reading")
        return False, None , None

# Perform inference
def run_inference(session, img_tensor):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: img_tensor})
    return outputs[0]

import os

# Enhanced error handling for OCR and image reading
# Draw bounding boxes on the image and run OCR
# Draw bounding boxes on the image and run OCR


# Function to calculate Intersection over Union (IoU)

def iou(box1, box2):
    global camera_name
    # Coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Area of intersection
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Area of both the boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Union area
    union_area = box1_area + box2_area - inter_area

    # IoU
    return inter_area / union_area

# Function to apply Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, iou_threshold):
    global camera_name
    print("NMA")
    if len(boxes) == 0:
        return []

    # Sort the boxes by confidence score in descending order
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    # List to hold the final bounding boxes after NMS
    nms_boxes = []

    while boxes:
        # Take the box with the highest confidence
        best_box = boxes.pop(0)
        nms_boxes.append(best_box)

        # Filter out boxes that have a high IoU with the best box
        boxes = [box for box in boxes if iou(best_box, box) < iou_threshold]

    return nms_boxes

def draw_boxes_and_ocr(image, boxes, class_names, event_id):
    global camera_name, car_bbox
    print("[Number-plate-detection] - Drawing boxes and OCR")
    file_path = os.path.join(CLIPS_PATH, f"{camera_name}-{event_id}-number-plate-detection.png")
    file_path_list = []
    best_match_name_list = []
    ocr_list = []

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Assuming the bounding box coordinates are originally scaled to 640x640
    scaled_width, scaled_height = 640, 640
    print(f"Original image dimensions: {original_width}x{original_height}")

    for idx, (x1, y1, x2, y2, conf, class_id) in enumerate(boxes):
        # Scale the bounding box coordinates back to the original image size
        x1 = int(x1 * original_width / scaled_width)
        x2 = int(x2 * original_width / scaled_width)
        y1 = int(y1 * original_height / scaled_height)
        y2 = int(y2 * original_height / scaled_height)
        print(f"[Number-plate-detection] - Box {idx} scaled coordinates: x1={x1}, x2={x2}, y1={y1}, y2={y2}")

        # Ensure the bounding box is valid
        if x1 < 0 or x2 > original_width or y1 < 0 or y2 > original_height:
            print(f"Invalid bounding box coordinates for box {idx}, skipping.")
            continue

        x1_new, y1_new, x2_new, y2_new = car_bbox
        print("x1: ", x1_new)
        print("y1: ", y1_new)
        # Check if the old bounding box is inside the new bounding bo
        if x1 >= x1_new and y1 >= y1_new and x2 <= x2_new and y2 <= y2_new:
            print(f"[Number-plate-detection] - Old box is inside the new bounding box.")
        else:
            print(f"[Number-plate-detection] - Old box is NOT inside the new bounding box.")
            continue

        # Draw rectangle around detected object
        label = f'{class_names[class_id]} {conf:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Crop the image to the bounding box
        plate_img = image[y1:y2, x1:x2]

        if plate_img is None or plate_img.size == 0:
            print(f"[Number-plate-detection] - Error: Invalid cropped image for box {idx}, skipping.")
            continue

        # Run OCR and handle errors
        try:
            ocr_result = ocr.ocr(plate_img)
            if not ocr_result or len(ocr_result[0]) == 0:
                print(f"[Number-plate-detection] - No text detected by OCR for box {idx}, skipping.")
                continue
        except Exception as e:
            print(f"[Number-plate-detection] - Error during OCR for box {idx}: {e}")
            continue

        # Extract recognized text from OCR result
        ocr_text = ' '.join([res[1][0] for res in ocr_result[0]]) if ocr_result else None
        print(f"[Number-plate-detection] - OCR Result for box {idx} (number plate): {ocr_text}")

        # Match OCR text to car labels
        if ocr_text:
            best_match_name = "Unknown car"
            best_similarity = 0

            for car_dict in car_label_list:
                for car_name, license_number in car_dict.items():
                    if license_number:
                        similarity = jellyfish.jaro_winkler_similarity(ocr_text, license_number)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_name = car_name

            if best_similarity > 0.7:
                print(f"[Number-plate-detection] - Best match for box {idx}: {best_match_name} with similarity {best_similarity}")
                if best_match_name not in best_match_name_list:
                    best_match_name_list.append(best_match_name)
                    ocr_list.append(ocr_text)
            else:
                print(f"[Number-plate-detection] - No match found for box {idx} (unknown car)")
                best_match_name_list.append(str(ocr_text))
                ocr_list.append(ocr_text)

    # Save the image after processing all detections
    cv2.imwrite(file_path, image)
    print(f"[Number-plate-detection] - Image saved at {file_path}")
    file_path_list.append(file_path)
    print("[Number-plate-detection] - Best matches: ", best_match_name_list)
    print("[Number-plate-detection] - Ocr list: ", ocr_list)
    print("[Number-plate-detection] - Path: ", file_path_list)
    return file_path_list, best_match_name_list, ocr_list


# Load and preprocess image
def preprocess_image(img, img_size=640):
    global camera_name
    print("[Number-plate-detection] - Pre-processing image")
    img_resized = cv2.resize(img, (img_size, img_size))
    # Convert image to a tensor (NCHW format) and normalize to [0, 1]
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))  # Change from HWC to CHW
    img_tensor = np.expand_dims(img_tensor, axis=0)    # Add batch dimension
    return img, img_tensor

# Post-process output to extract bounding boxes, confidences, and class IDs with NMS
def post_process(output, conf_threshold, iou_threshold):
    global camera_name
    boxes = []
    class_id_1_count = 0  # Counter for class_id 1
    print("[Number-plate-detection] - Post-processing image")
    number_plate_found = False

    # Loop through each detection
    for detection in output[0]:
        confidence = detection[4]  # confidence score

        if confidence >= conf_threshold:
            # Extract coordinates directly without scaling
            x_center, y_center, width, height = detection[0:4]
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Get the class with the highest score from class probabilities
            class_id = np.argmax(detection[5:])

            if class_id == 1:  # Check if it's class_id 1
                number_plate_found = True
                class_id_1_count += 1  # Increment counter
                # Append bounding box and other details to the list
                boxes.append((x1, y1, x2, y2, confidence, class_id))

    # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    final_boxes = non_max_suppression(boxes, iou_threshold)

    print(f"[Number-plate-detection] - Total Class ID 1 detections (after NMS): {len(final_boxes)}")

    print("[Number-plate-detection] - Number plate found: ", number_plate_found)
    return number_plate_found, final_boxes

def main_processing(img, event_id, event_data):
    global camera_name
    print("[Number-plate-detection] - Started main processing")
    global session
    global conf_threshold
    #preprocess image
    original_image, img_tensor = preprocess_image(img)
    # Run inference
    output = run_inference(session, img_tensor)
    # Post-process output to get bounding boxes and see if number plate was detected
    number_plate_result, boxes = post_process(output, conf_threshold, 0.8)
    # Draw boxes on the original image
    if(number_plate_result):
        file_path, best_match_name, ocr_text = draw_boxes_and_ocr(original_image, boxes, class_names, event_id)
        print("[Number-plate-detection] - Number plate results found after ocr")
        return file_path, best_match_name, ocr_text
    else:
        file_path = os.path.join(CLIPS_PATH, f"{camera_name}-{event_id}-number-plate-detection.png")
        print("[Number-plate-detection] - Ocr detection failed")
        best_match_name = None
        ocr_text = None
        return file_path, best_match_name, ocr_text
    

# Utility function to ensure the 'event' table exists in the given database
def ensure_event_table_exists(db_path):
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        cursor = connection.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='event';
        """)
        result = cursor.fetchone()
        if result is None:
            # Table does not exist, set it up
            setup_database(db_path)
            print(f"[Number-plate-processing] - Created 'event' table in {db_path}.")





def setup_database(db_path):
    print("Setting up database")
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        cursor = connection.cursor()
        cursor.execute(EVENT_TABLE_SCHEMA)
        cursor.execute(PROCESSING_TABLE_SCHEMA)
        connection.commit()

def update_database(event_id, event_data, out_image_path=None, video_path=None, recognized_name=None, table_name='events', status=None, status_time=None):
    loitery_string = "None"
    additional_info = "No"
    #entry into the database
    start_time = time.time()
    print("[Number-plate-processing] -  inside update database")
    ensure_event_table_exists(EVENTS_DB_PATH)

    while True:
        try:
            print("[Number-plate-processing] -  Copying frigate db")
            with sqlite3.connect(FRIGATE_DB_PATH) as frigate_db_con:
                cursor = frigate_db_con.cursor()
                cursor.execute("SELECT id, label, camera, start_time, end_time, thumbnail FROM event WHERE id = ?", (event_id,))
                event_data = cursor.fetchone()
                if event_data and len(event_data) == 6:
                    break
        except sqlite3.Error as e:
            logging.error(f"Error accessing Frigate database: {e}")
        if time.time() - start_time > 30:
            return
        time.sleep(1)

    if table_name == 'events':
        if event_data and len(event_data) == 6:
            print("[Number-plate-processing] -  Event data is length 6")
            start_time = time.time()
            while True:
                try:
                    with sqlite3.connect(EVENTS_DB_PATH) as events_db_con:
                        events_db_con.execute("PRAGMA foreign_keys = ON;")
                        #setup_database(events_db_con)
                        cursor = events_db_con.cursor()

                        cursor.execute("""
                            INSERT INTO event (
                                id, label, camera, start_time, end_time, thumbnail,
                                sub_label, snapshot_path, video_path, loitering, additional_info
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT (id) DO UPDATE SET
                                sub_label = excluded.sub_label,
                                snapshot_path = excluded.snapshot_path,
                                video_path = excluded.video_path,
                                loitering = excluded.loitering,
                                additional_info = excluded.additional_info
                        """, (
                            event_data[0], event_data[1], event_data[2], event_data[3], event_data[4], event_data[5],
                            recognized_name, out_image_path, video_path, loitery_string, additional_info
                        ))

                        events_db_con.commit()
                        print("[Number-plate-processing] - Data inserted/updated successfully in the events database.")
                        break
                except sqlite3.Error as e:
                    logging.error(f"Error accessing Events database: {e}")
                if time.time() - start_time > 30:
                    return
                time.sleep(1)

            current_time = time.time()
            period = current_time - start_time
            logging.info(f"Updating database took {period} seconds.")
        else:
            logging.error("File was not created in time.")
    else:
        if event_data and len(event_data) == 6:
            print("[Number-plate-processing] -  Event data is length 6")
            start_time = time.time()
            while True:
                try:
                    with sqlite3.connect(EVENTS_DB_PATH) as events_db_con:
                        events_db_con.execute("PRAGMA foreign_keys = ON;")
                        #setup_database(events_db_con)
                        cursor = events_db_con.cursor()

                        cursor.execute("""
                            INSERT INTO processing (
                                event_id,
                                status,
                                status_time
                            ) VALUES (?, ?, ?)
                        """, (
                            event_data[0], status, status_time
                        ))

                        events_db_con.commit()
                        print("[Number-plate-processing] - Data inserted/updated successfully in the processing table of events database.")
                        break
                except sqlite3.Error as e:
                    logging.error(f"Error accessing Events database: {e}")
                if time.time() - start_time > 30:
                    return
                time.sleep(1)

            current_time = time.time()
            period = current_time - start_time
            logging.info(f"Updating database took {period} seconds.")
        else:
            logging.error("File was not created in time.")
    
# Mock function to initialize YOLO model
def initialize_model(onnx_model_path):
    """Initialize the YOLO model."""
    model = YOLO(onnx_model_path, task='detect')
    return model



def process_video(event_id, event_data, video_path, minimum_duration=2):
    """
    Process a video, track car directions, and update the database.

    Args:
        event_id (str): Unique identifier for the event.
        event_data (dict): Additional data related to the event.
        video_path (str): Path to the video file.
    """
    # Initialize YOLO model
    model = initialize_model(YOLO_MODEL)  # Replace with your YOLO model path
    cap = cv2.VideoCapture(video_path)
    car_directions = {}
    frame_counter = 0
    initial_frames = 10
    update_interval = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no more frames

        frame_counter += 1

        # Run YOLO detection with tracking (filter for car class)
        results = model.track(source=frame, stream=True, persist=True, classes=[2])

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else -1

                # Calculate bounding box size (area)
                bbox_size = (x2 - x1) * (y2 - y1)

                # Initialize car data
                if track_id not in car_directions:
                    car_directions[track_id] = {
                        'initial_sizes': [],
                        'final_sizes': [],
                        'initial_average_size': None,
                        'final_average_size': None,
                        'direction': "Unknown",
                        'last_direction': None,
                        'last_update_time': None
                    }
                car_data = car_directions[track_id]

                # Collect initial frame sizes
                if car_data['initial_average_size'] is None:
                    car_data['initial_sizes'].append(bbox_size)
                    if len(car_data['initial_sizes']) == 7:
                        sizes = sorted(car_data['initial_sizes'])[1:-1]  # Trim outliers
                        car_data['initial_average_size'] = sum(sizes) / len(sizes)

                # Collect final frame sizes
                elif car_data['final_average_size'] is None:
                    car_data['final_sizes'].append(bbox_size)
                    if len(car_data['final_sizes']) == 23:
                        car_data['final_sizes'] = car_data['final_sizes'][16:]  # Keep last 7 sizes
                        sizes = sorted(car_data['final_sizes'])[1:-1]  # Trim outliers
                        car_data['final_average_size'] = sum(sizes) / len(sizes)

                # Update state every 30 frames
                elif (frame_counter - initial_frames) % update_interval == 0 and car_data['final_average_size'] is not None:
                    size_diff = car_data['final_average_size'] - car_data['initial_average_size']
                    margin = 0.02 * car_data['initial_average_size']

                    # Decide direction
                    if abs(size_diff) <= margin:
                        new_direction = 'Parked'
                    else:
                        new_direction = 'Approaching' if size_diff > margin else 'Departing'

                    # Update only if direction changes and stable
                    current_time = frame_counter / cap.get(cv2.CAP_PROP_FPS)
                    if (new_direction != car_data['last_direction'] and
                            (car_data['last_update_time'] is None or (current_time - car_data['last_update_time']) >= minimum_duration)):
                        car_data['last_direction'] = new_direction
                        car_data['last_update_time'] = current_time
                        update_database(event_id, event_data, table_name='processing', status=new_direction, status_time=round(current_time, 2))

                    # Reset for next state calculation
                    car_data['initial_average_size'] = None
                    car_data['final_average_size'] = None
                    car_data['initial_sizes'] = []
                    car_data['final_sizes'] = []

                    # Update the dictionary explicitly
                    car_directions[track_id] = car_data

    cap.release()