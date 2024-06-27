import cv2
import numpy as np

def load_yolo_model(yolo_cfg, yolo_weights, yolo_names):
    net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open(yolo_names, 'r') as f:
        classes = f.read().strip().split('\n')

    return net, output_layers, classes

def detect_phones(net, output_layers, classes, frame):
    small_frame = cv2.resize(frame, (320, 320))
    blob = cv2.dnn.blobFromImage(small_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    phone_boxes = []
    confidences = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "cell phone":
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                phone_boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(phone_boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    return phone_boxes, indices
