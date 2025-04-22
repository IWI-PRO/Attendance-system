import os
import cv2
import dlib
import numpy as np
import pickle
import logging
from detection import FaceDetector  # Importing FaceDetector class from detection.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceEncoder:
    def __init__(self, face_dir="image", embedding_file="embeddings.pkl", detector_type="hog"):
        """
        Initializes the FaceEncoder with a directory containing face images.
        face_dir: Path to the folder containing images.
        embedding_file: Path to save the embeddings file.
        detector_type: Face detection method ('hog' or 'cnn').
        """
        self.face_dir = face_dir
        self.embedding_file = embedding_file
        self.detector = FaceDetector(image_folder=face_dir, detector_type=detector_type)

        try:
            self.shape_predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
            self.face_recognizer = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")
            logging.info("Dlib models loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading dlib models: {e}")
            raise

    def get_face_embedding_dlib(self, img, face):
        """
        Generate a 128-D face embedding for a detected face.
        img: Input image.
        face: dlib rectangle object representing the detected face.
        return: Face embedding as a NumPy array.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            landmarks = self.shape_predictor(gray, face)
            face_embedding = self.face_recognizer.compute_face_descriptor(img, landmarks)
            return np.array(face_embedding)
        except Exception as e:
            logging.error(f"Error generating face embedding: {e}")
            return None

    def process_images(self):
        """
        Process images in the given directory, extract face embeddings, and save them.
        """
        embeddings_dict = {}
        
        for foldername in sorted(os.listdir(self.face_dir)):
            augment_folder = os.path.join(self.face_dir, foldername, "augment")
            if not os.path.isdir(augment_folder):
                continue

            parts = foldername.split('-')
            if len(parts) < 3:
                logging.warning(f"Skipping invalid folder format: {foldername}")
                continue

            employee_id = parts[1]  # Extract ID
            employee_name = ' '.join(parts[2:])  # Extract name
            logging.info(f"Processing Employee: {employee_name} (ID: {employee_id})")

            for filename in sorted(os.listdir(augment_folder)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(augment_folder, filename)
                    
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            logging.warning(f"Skipping unreadable file: {filename}")
                            continue

                        faces = self.detector.face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
                        if not faces:
                            logging.info(f"No face detected in {filename}, skipping...")
                            continue

                        for i, face in enumerate(faces):
                            embedding = self.get_face_embedding_dlib(img, face)
                            if embedding is not None:
                                if employee_id not in embeddings_dict:
                                    embeddings_dict[employee_id] = {"name": employee_name, "embeddings": []}
                                embeddings_dict[employee_id]["embeddings"].append(embedding)
                                logging.info(f"Processed {filename}: Face {i} encoded.")
                    except Exception as e:
                        logging.error(f"Error processing file {filename}: {e}")
                        continue

        # Save embeddings to a pickle file
        try:
            with open(self.embedding_file, "wb") as f:
                pickle.dump(embeddings_dict, f)
            logging.info(f"Embeddings saved successfully to {self.embedding_file}")
        except Exception as e:
            logging.error(f"Error saving embeddings: {e}")

if __name__ == "__main__":
    try:
        encoder = FaceEncoder()
        encoder.process_images()
    except Exception as e:
        logging.critical(f"Critical error in execution: {e}")
