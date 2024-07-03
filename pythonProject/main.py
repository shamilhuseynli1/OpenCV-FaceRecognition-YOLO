import cv2
import face_recognition
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import os
from Scripts.face_recognition import load_images_and_encode_faces
from Scripts.yolo import load_yolo_model, detect_phones
from Scripts.deepface import analyze_emotion, process_face

def recognize_faces_and_log_shifts():
    # Load known face encodings and labels
    faces_dir = 'Faces'
    known_face_encodings, labels = load_images_and_encode_faces(faces_dir)

    # Load YOLO model for phone detection
    yolo_cfg = 'YOLO/yolov3.cfg'
    yolo_weights = 'YOLO/yolov3.weights'
    yolo_names = 'YOLO/coco.names'
    net, output_layers, classes = load_yolo_model(yolo_cfg, yolo_weights, yolo_names)

    # Initialize video capture from the default camera
    cap = cv2.VideoCapture(0)
    frame_count = 0
    process_frame_rate = 15  # Process every 15th frame
    shift_log = defaultdict(list)  # Log of detected faces and their shift timings
    inactivity_threshold = timedelta(seconds=10)  # Time threshold to consider inactivity
    restart_threshold = timedelta(seconds=10)  # Time threshold to restart a shift
    label_duration = timedelta(seconds=3)  # Duration to display labels on screen
    last_detected_faces = {}  # Store last detected faces and their details
    emotion_analysis_results = {}  # Store results of emotion analysis

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = datetime.now()  # Get the current time at the start of each loop iteration

        # Process frame at a defined rate
        if frame_count % process_frame_rate == 0:
            face_locations = face_recognition.face_locations(frame)  # Detect face locations
            face_encodings = face_recognition.face_encodings(frame, face_locations)  # Get face encodings

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)  # Compare face with known faces
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = labels[match_index]

                # Log start time if not already logged
                if name != "Unknown":
                    if not shift_log[name] or 'end_time' in shift_log[name][-1]:
                        if not shift_log[name] or current_time - shift_log[name][-1]['end_time'] > restart_threshold:
                            shift_log[name].append({'start_time': current_time})
                            print(f"{name} shift started at {current_time}")
                    if shift_log[name] and 'end_time' not in shift_log[name][-1]:
                        shift_log[name][-1]['last_seen'] = current_time

                # Update last detected face information
                last_detected_faces[name] = {
                    'time': current_time,
                    'location': (top, right, bottom, left)
                }

                if name != "Unknown":
                    face_roi_resized = process_face(frame, top, right, bottom, left)
                    threading.Thread(target=analyze_emotion, args=(face_roi_resized, name, current_time, emotion_analysis_results)).start()

        # Detect phones using YOLO model
        phone_boxes, indices = detect_phones(net, output_layers, classes, frame)
        phone_found = False

        if len(indices) > 0:
            for i in indices.flatten():
                box = phone_boxes[i]
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Phone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                phone_found = True

                # Check if phone is within any face boundary
                for name, info in last_detected_faces.items():
                    top, right, bottom, left = info['location']
                    if x >= left and x + w <= right and y >= top and y + h <= bottom:
                        print(f"{name} is using a phone at {current_time}")
                        break

        # Check for inactivity and log end time
        for name, shifts in shift_log.items():
            if shifts and 'last_seen' in shifts[-1] and 'end_time' not in shifts[-1]:
                if current_time - shifts[-1]['last_seen'] > inactivity_threshold:
                    shifts[-1]['end_time'] = shifts[-1]['last_seen'] + inactivity_threshold
                    print(f"{name} shift ended at {shifts[-1]['end_time']}")
                    start_time = shifts[-1]['start_time']
                    end_time = shifts[-1]['end_time']
                    total_time = end_time - start_time
                    print(f"{name} total shift time was {total_time}")

        # Display last detected faces
        for name, info in last_detected_faces.items():
            if current_time - info['time'] < label_duration:
                top, right, bottom, left = info['location']
                emotion_text = emotion_analysis_results.get(name, {}).get('emotion', '')

                # Draw rectangle around the face and label it
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                if emotion_text:
                    cv2.putText(frame, emotion_text, (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.imshow('Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_faces_and_log_shifts()
