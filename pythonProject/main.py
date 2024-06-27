import cv2
import face_recognition
from datetime import datetime, timedelta
from collections import defaultdict
from Scripts.image_processing import load_images_and_encode_faces
from Scripts.yolo import load_yolo_model, detect_phones

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
    shift_log = defaultdict(list)
    inactivity_threshold = timedelta(seconds=10)  # Define inactivity threshold
    restart_threshold = timedelta(seconds=10)  # Define restart threshold after end time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = datetime.now()  # Get the current time at the start of each loop iteration

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
                    if not shift_log[name] or 'end_time' in shift_log[name][-1]:
                        # Check if enough time has passed since the last shift ended
                        if not shift_log[name] or current_time - shift_log[name][-1]['end_time'] > restart_threshold:
                            shift_log[name].append({'start_time': current_time})
                            print(f"{name} shift started at {current_time}")
                    # Update last seen time whenever the face is detected
                    if shift_log[name] and 'end_time' not in shift_log[name][-1]:
                        shift_log[name][-1]['last_seen'] = current_time

                # Draw rectangle around the face and label it
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
        for name, shifts in shift_log.items():
            if shifts and 'last_seen' in shifts[-1] and 'end_time' not in shifts[-1]:
                # Calculate the difference between current time and last seen time
                if current_time - shifts[-1]['last_seen'] > inactivity_threshold:
                    shifts[-1]['end_time'] = shifts[-1]['last_seen'] + inactivity_threshold  # Log the end time
                    print(f"{name} shift ended at {shifts[-1]['end_time']}")

        cv2.imshow('Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Print shift logs
    for name, shifts in shift_log.items():
        for i, times in enumerate(shifts):
            print(f"Shift {i + 1} for {name}:")
            print(f"  Start: {times.get('start_time')}")
            print(f"  Last Seen: {times.get('last_seen')}")
            end_time = times.get('end_time', times.get('last_seen') + inactivity_threshold)
            print(f"  End: {end_time}")
            total_time = end_time - times.get('start_time')
            print(f"  Total logged time: {total_time}")

if __name__ == '__main__':
    recognize_faces_and_log_shifts()
