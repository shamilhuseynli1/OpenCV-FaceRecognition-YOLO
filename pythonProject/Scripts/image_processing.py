import os
import face_recognition
from PIL import UnidentifiedImageError

def load_images_and_encode_faces(faces_dir):
    known_face_encodings = []
    labels = []

    # List of valid image file extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for img_name in os.listdir(faces_dir):
        if not img_name.lower().endswith(valid_extensions):
            print(f"Skipping non-image file: {img_name}")
            continue

        img_path = os.path.join(faces_dir, img_name)

        try:
            img = face_recognition.load_image_file(img_path)
            img_encoding = face_recognition.face_encodings(img)[0]

            known_face_encodings.append(img_encoding)
            labels.append(os.path.splitext(img_name)[0])

        except UnidentifiedImageError:
            print(f"Skipping file that cannot be identified as an image: {img_path}")
        except IndexError:
            print(f"No face found in the image: {img_path}")
        except Exception as e:
            print(f"Skipping file due to an error: {img_path}, Error: {str(e)}")

    return known_face_encodings, labels
