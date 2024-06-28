from deepface import DeepFace
import cv2

# Custom labels for emotions
emotion_labels = {
    "angry": "Angry",
    "fear": "Scared",
    "happy": "Joyful",
    "sad": "Unhappy",
    "surprise": "Surprised",
    "neutral": "Calm"
}

def analyze_emotion(face_roi, name, current_time, emotion_analysis_results):
    try:
        # Analyze emotion
        emotion_analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        dominant_emotion = emotion_analysis[0]['dominant_emotion']
        emotion_text = emotion_labels.get(dominant_emotion, dominant_emotion)
        emotion_analysis_results[name] = {
            'emotion': emotion_text,
            'time': current_time
        }
    except Exception as e:
        print(f"Emotion detection error: {e}")

def process_face(frame, top, right, bottom, left):
    face_roi = frame[top:bottom, left:right]
    face_roi_resized = cv2.resize(face_roi, (48, 48))
    return face_roi_resized
