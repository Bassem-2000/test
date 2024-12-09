import paho.mqtt.client as mqtt_client
import logging
import json
import sqlite3
import time
import os
from datetime import datetime
import threading
from number_plate_detection import generate_number_plate_bbox
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from queue import Queue
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
event_id = False
event_data = []
number_plate_processing_ids = []
thread_inprocess = False
number_plate_processed_ids =[]
camera_name = "None"
from constants import (CLIPS_PATH, RECORDINGS_PATH, FRIGATE_SERVER_ADDRESS, MQTT_TOPIC, NUMBER_PLATE_DATA_TOPIC)
from constants import *
# Paths for heatmap images

# Callback when the client receives a connection response from the broker
def connect_number_plate(client, userdata, flags, rc):
    print("[Number-plate-processing] -  Connected with result code " + str(rc))
    # Subscribe to the 'number_plate_recognition' topic
    client.subscribe(MQTT_TOPIC)

# The callback for when a PUBLISH message is received from the broker.
def on_message_number_plate(client, userdata, msg):
    print("Received data from MQTT")
    global event_id, event_data, number_plate_processing_ids, thread_inprocess, number_plate_processed_ids, camera_name 

    payload = msg.payload.decode()
    # Decode the received message
    data = json.loads(payload)
    event_id = data.get('after', {}).get('id', None)
    # event_id = data.get('event_id', None)
    event_data = data['after']
    
    cam_name = data.get('after', {}).get('camera', None)
    camera_name = cam_name

    print("Camera name is: ", camera_name)

    if event_id and ('car' in data.get('after', {}).get('label', None)) and event_data.get('has_snapshot'):
        if event_id not in number_plate_processing_ids and not thread_inprocess and event_id not in number_plate_processed_ids:
            thread_inprocess = True
            number_plate_processing_ids.append(event_id)
            print(f"Event ID {event_id} added to number plate processing and being processed immediately")
            # Process the event immediately (no threads)
            thread = threading.Thread(target=number_plate_process_event, args=(event_id, event_data))
            thread.start()
        else:
            if(thread_inprocess):
                print(f"Event ID {event_id} thread already running")
            elif event_id in number_plate_processing_ids:
                print(f"Event ID {event_id} is already in processing ids")
            elif event_id in number_plate_processed_ids:
                print(f"Event ID {event_id} is already in processed ids")
    elif not event_data.get('has_snapshot'):
        print(f"Event ID {event_id} has invalid snapshot")

# Function to process the event and handle face recognition
def number_plate_process_event(event_id, event_data):
    global thread_inprocess, number_plate_processing_ids, number_plate_processed_ids, camera_name 
    print("[Number-plate-processing] -  Inside number plate thread for: ", event_id)
    global date_format, thread_inprocess
    file_path = os.path.join(CLIPS_PATH, f"{camera_name}-{event_id}-number-plate.png")
    snapshot_image = fetch_best_snapshot(event_id,file_path)
    print("Saving snapshot")
    save_snapshot_image(snapshot_image, file_path)
    print("Snapshot saved")
    date_format = str(datetime.fromtimestamp(event_data['frame_time']))
    if wait_for_file_creation(file_path):
        file_path, best_match_name, ocr_text = generate_number_plate_bbox(event_id, event_data, camera_name)
        print("[Number-plate-processing] -  Best matches: ", best_match_name)
        print("[Number-plate-processing] -  Ocr list: ", ocr_text)
        print("[Number-plate-processing] -  Path: ", file_path)
        print("[Number-plate-processing] -  Number plate result received")
        date_format = str(datetime.fromtimestamp(event_data['frame_time']))
        video_path = get_video_path(date_format)
        if best_match_name:
            print("publishing number plate data")
            # Publish best_match_name and frame_time on MQTT topi
            mqtt_data = {
                    "best_match_name": best_match_name,
                    "frame_time": event_data['frame_time']
                    }
            mqtt_topic = NUMBER_PLATE_DATA_TOPIC
            client.publish(mqtt_topic, json.dumps(mqtt_data))
            for i in range(len(best_match_name)):
                recognized_name = str(best_match_name[i])
                print("[Number-plate-processing] -  Calling Database update")
                update_database(event_id, event_data, file_path[0], video_path, recognized_name)
                number_plate_processed_ids.append(event_id)

        else:
            print("[Number-plate-processing] -  Number plate not detected for event ID")
            if event_id in number_plate_processing_ids:
                number_plate_processing_ids.remove(event_id)
            if event_id in number_plate_processed_ids:
                number_plate_processed_ids.remove(event_id)
            print("[Number-plate-processing] -  Number plate not event id removed as not detected: ", event_id)

    else:
        print("[Number-plate-processing] -  Not sending any data as file not read")
    thread_inprocess = False

def get_video_path(date_format):
    global camera_name 
    folder_path = os.path.join(RECORDINGS_PATH, date_format[:10], '00')
    creation_hour = get_folder_creation_hour(folder_path)
    if isinstance(creation_hour, str):
        print(creation_hour)  # Log the error message
        creation_hour = 0  # Default to 0 if folder does not exist
    subfolder_num = int(date_format[11:13]) - int(creation_hour)
    subfolder_num_str = str(subfolder_num).zfill(2)
    video_path = os.path.join(RECORDINGS_PATH, date_format[:10], subfolder_num_str, camera_name, f"{date_format[14:16]}.{date_format[17:19]}.mp4")
    return video_path

def get_folder_creation_hour(folder_path):
    try:
        creation_time = os.path.getctime(folder_path)
        creation_time_struct = time.localtime(creation_time)
        creation_hour = creation_time_struct.tm_hour
        return creation_hour
    except FileNotFoundError:
        return "The specified folder does not exist."

def fetch_best_snapshot(event_id, base_url= FRIGATE_SERVER_ADDRESS):
    # Construct the URL for accessing the best snapshot
    snapshot_url = f"{FRIGATE_SERVER_ADDRESS}/api/events/{event_id}/snapshot.jpg"
    # Make the HTTP request to fetch the image
    response = requests.get(snapshot_url)
    if response.status_code == 200:
        # Load the response content as an image
        print("[Number-plate-processing] -  Snapshot saved for event id: ", event_id)
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print(f"Failed to fetch snapshot. Status code: {response.status_code}")
        return None

def save_snapshot_image(image, file_path):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Save the image
        image.save(file_path)
        print(f"Snapshot saved at: {file_path}")
    except Exception as err:
        print(f"Failed to save image: {err}")

def wait_for_file_creation(file_path, timeout=10, check_interval=0.5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    f.read()
                return True
            except IOError:
                pass
        time.sleep(check_interval)
    logging.error(f"Timeout reached. File not found or not ready: {file_path}")
    return False


# Utility function to ensure the 'event' table exists in the given database
def ensure_event_table_exists(db_path):
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='event';
        """)
        result = cursor.fetchone()
        if result is None:
            # Table does not exist, set it up
            setup_database(db_path)
            print(f"[Number-plate-processing] - Created 'event' table in {db_path}.")


def update_database(event_id, event_data, out_image_path, video_path, recognized_name):
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

    if event_data and len(event_data) == 6:
        print("[Number-plate-processing] -  Event data is length 6")
        start_time = time.time()
        while True:
            try:
                with sqlite3.connect(EVENTS_DB_PATH) as events_db_con:
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

# Utility functions (update_heatmap, apply_heatmap_pixel_by_pixel, save_heatmap, etc.)
# These are the same as in your original code
# Function to set up the database tables if they do not exist

def setup_database(db_path):
    print("Setting up database")
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(EVENT_TABLE_SCHEMA)
        connection.commit()

# Start the MQTT client
def start_client(client):
    client.loop_forever()

if __name__ == "__main__":
    # Number Plate Recognition client
    client = mqtt_client.Client()
    client.on_connect = connect_number_plate
    client.on_message = on_message_number_plate
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(BROKER_HOST, BROKER_PORT, 60)

    print("Starting number plate recognition node")

    # Start the client
    start_client(client)