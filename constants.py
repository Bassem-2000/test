# MQTT Constants  
BROKER_HOST = "jupyter-PoC"  
BROKER_PORT = 1883  
MQTT_USERNAME = "admin"  
MQTT_PASSWORD = "1BeachHouse@2023"  
MQTT_TOPIC = "frigate/events"  

#topic on which number plate data will be published via MQTT
NUMBER_PLATE_DATA_TOPIC = "frigate/number_plate_data"
LICENCE_RECOGNITION_DETECTIONS_TOPIC = "licence_recognition/detections"


# Paths  


FRIGATE_SERVER_ADDRESS = "http://jupyter-PoC:5000"

#FRIGATE_SERVER_ADDRESS = "http://" + BROKER_HOST + ":5000"

FRIGATE_DB_PATH = "/home/admin/config/frigate.db"  
EVENTS_DB_PATH = "/home/admin/config/events.db"  
CLIPS_PATH = "/home/admin/storage/clips/"  
RECORDINGS_PATH = "/home/admin/storage/recordings/"  

#Number Plate detection
MODEL_PATH = "best_v3_final.onnx"
NUMBER_PLATE_DETECTION_SCORE = 0.55
CSV_FILE_PATH = 'input.csv'

# Database Schema  
EVENT_TABLE_SCHEMA = """  
    CREATE TABLE IF NOT EXISTS event (  
        id TEXT PRIMARY KEY,   
        label TEXT,   
        camera TEXT,   
        start_time DATETIME,  
        end_time DATETIME,  
        thumbnail TEXT,  
        sub_label TEXT,  
        snapshot_path TEXT,  
        video_path TEXT,
        loitering TEXT,
        additional_info TEXT,
        face_detection_score TEXT,
        parcel_status TEXT
        )
        """
