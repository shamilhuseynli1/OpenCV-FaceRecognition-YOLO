import cv2
import face_recognition
from datetime import datetime, timedelta
from collections import defaultdict
from Scripts.image_processing import load_images_and_encode_faces
from Scripts.yolo import load_yolo_model, detect_phones
import numpy as np
import pandas as pd

def recognize_faces_and_log_shifts():
    faces_dir = 'Faces'
    known_face_encodings, labels = load_images_and_encode_faces(faces_dir)

    yolo_cfg = 'YOLO/yolov3.cfg'
    yolo_weights = 'YOLO/yolov3.weights'
    yolo_names = 'YOLO/coco.names'

    net, output_layers, classes = load_yolo_model(yolo_cfg, yolo_weights, yolo_names)

    cap = cv2.VideoCapture(0)
    frame_count = 0
    process_frame_rate = 10
    shift_log = defaultdict(dict)
    inactivity_threshold = timedelta(seconds=6)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = datetime.now()

        if frame_count % process_frame_rate == 0:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = labels[match_index]

                # Log start time if not already logged
                if name != "Unknown":
                    if 'start_time' not in shift_log[name]:
                        shift_log[name]['start_time'] = current_time
                        print(f"{name} shift started at {current_time}")
                    shift_log[name]['last_seen'] = current_time

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        phone_boxes, indices = detect_phones(net, output_layers, classes, frame)
        phone_found = False

        if len(indices) > 0:
            for i in indices.flatten():
                box = phone_boxes[i]
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Phone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                phone_found = True

        # Check for inactivity and log end time
        for name, times in list(shift_log.items()):
            if 'last_seen' in times and current_time - times['last_seen'] > inactivity_threshold:
                if 'end_time' not in times:
                    shift_log[name]['end_time'] = current_time
                    print(f"{name} shift ended at {current_time}")

        cv2.imshow('Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Prepare shift logs for saving
    shift_data = []
    for name, times in shift_log.items():
        shift_data.append({
            'Name': name,
            'Start Time': times.get('start_time'),
            'Last Seen': times.get('last_seen'),
            'End Time': times.get('end_time', 'Still active')
        })

    # Create a DataFrame
    df = pd.DataFrame(shift_data)

    # Save the DataFrame to an Excel file
    output_file = 'shift_logs.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Shift logs saved to {output_file}")

if __name__ == '__main__':
    recognize_faces_and_log_shifts()