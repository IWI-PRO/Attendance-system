import os
import cv2
import dlib
import numpy as np
import pickle
import logging

class FaceDetector:
    def __init__(self, image_folder="image", detector_type="hog"):
        """
        Initializes the face detector with paths and configurations.

        Parameters:
        - image_folder (str): Root folder containing subdirectories with images.
        - detector_type (str): Type of detector to use ("hog" for faster processing, "cnn" for higher accuracy).
        """
        self.image_folder = image_folder

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        try:
            if detector_type == "cnn":
                self.face_detector = dlib.cnn_face_detection_model_v1('model/mmod_human_face_detector.dat')
            else:
                self.face_detector = dlib.get_frontal_face_detector()
        except Exception as e:
            logging.error(f"Error loading face detector: {e}")
            raise

        self.valid_extensions = (".jpg", ".jpeg", ".png")
        self.embeddings = {}

    def detect_faces(self):
        """
        Detects faces in images and stores embeddings in a pickle file.
        """
        try:
            for foldername in sorted(os.listdir(self.image_folder)):
                augment_folder = os.path.join(self.image_folder, foldername, "augment")

                if os.path.isdir(augment_folder):
                    for filename in sorted(os.listdir(augment_folder)):
                        if filename.lower().endswith(self.valid_extensions):
                            image_path = os.path.join(augment_folder, filename)

                            try:
                                img = cv2.imread(image_path)
                                if img is None:
                                    logging.warning(f"Skipping unreadable file: {filename}")
                                    continue

                                if len(img.shape) == 2:
                                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                faces = self.face_detector(gray, 1)

                                if len(faces) == 0:
                                    logging.info(f"No face detected in {filename}, skipping...")
                                    continue

                                for i, face in enumerate(faces):
                                    try:
                                        if isinstance(face, dlib.rectangle):
                                            x, y, w, h = face.left(), face.top(), face.width(), face.height()
                                        else:
                                            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()

                                        x, y, w, h = max(0, x), max(0, y), min(img.shape[1], w), min(img.shape[0], h)

                                        if x + w > img.shape[1] or y + h > img.shape[0]:
                                            logging.warning(f"Skipping face in {filename} due to invalid bounding box")
                                            continue

                                        face_roi = img[y:y+h, x:x+w]
                                        if face_roi.size == 0:
                                            logging.warning(f"Skipping face in {filename} due to empty ROI")
                                            continue

                                        id_name = os.path.splitext(filename)[0] + f"_face{i}"
                                        self.embeddings[id_name] = face_roi
                                    except Exception as e:
                                        logging.error(f"Error processing face in {filename}: {e}")
                                        continue
                            except Exception as e:
                                logging.error(f"Error processing file {filename}: {e}")
                                continue

            # Save embeddings as a pickle file
            try:
                with open("embeddings.pkl", "wb") as f:
                    pickle.dump(self.embeddings, f)
                logging.info("Face detection completed! Embeddings saved in 'embeddings.pkl'.")
            except Exception as e:
                logging.error(f"Error saving embeddings: {e}")

        except Exception as e:
            logging.error(f"Error during face detection: {e}")
            raise
