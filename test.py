import cv2
import dlib
import numpy as np
import pickle
import os
import json
import time
import logging
from datetime import datetime
from scipy.spatial.distance import cosine
from win32com.client import Dispatch

# Configure logging
logging.basicConfig(
    filename="face_recognition.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class FaceRecognition:
    def __init__(self, embedding_file="embeddings.pkl", attendance_dir="track", image_dir="image"):
        """
        Initializes the FaceRecognition system with required models and configurations.
        """
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
            self.face_recognizer = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise

        self.embedding_file = embedding_file
        self.attendance_dir = attendance_dir
        self.image_dir = image_dir
        self.threshold = 0.5
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        try:
            with open(self.embedding_file, "rb") as f:
                self.stored_embeddings = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading embeddings file: {e}")
            self.stored_embeddings = {}
        
        self.recorded_today = set()

    def speak(self, text):
        """Announce attendance using text-to-speech."""
        try:
            speaker = Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        except Exception as e:
            logging.error(f"Error in speech synthesis: {e}")

    def get_face_embedding_dlib(self, img):
        """Generate 128-D face embeddings for detected faces."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)
            if not faces:
                return [], [], []

            embeddings, face_boxes, confidences = [], [], []
            for face in faces:
                landmarks = self.shape_predictor(gray, face)
                face_embedding = np.array(self.face_recognizer.compute_face_descriptor(img, landmarks)).flatten()
                embeddings.append(face_embedding)
                face_boxes.append((face.left(), face.top(), face.right(), face.bottom()))
                confidence = getattr(face, 'confidence', 1.0)
                confidences.append(confidence)
            return embeddings, face_boxes, confidences
        except Exception as e:
            logging.error(f"Error in face encoding: {e}")
            return [], [], []

    def recognize_faces(self, frame):
        """Recognize faces and return annotated frame."""
        try:
            embeddings, face_boxes, _ = self.get_face_embedding_dlib(frame)
            recognized_faces = []
            
            for i, embedding in enumerate(embeddings):
                best_match_id, best_match_name, min_distance = "Unknown", "Unknown", self.threshold
                distances, labels = [], []
                for emp_id, data in self.stored_embeddings.items():
                    for stored_embedding in data["embeddings"]:
                        distance = np.linalg.norm(embedding - np.array(stored_embedding).flatten())
                        distances.append(distance)
                        labels.append(emp_id)
                
                if distances:
                    best_distance = min(distances)
                    if best_distance < min_distance:
                        best_match_id = labels[np.argmin(distances)]
                        best_match_name = self.stored_embeddings[best_match_id]["name"]
                
                x1, y1, x2, y2 = face_boxes[i]
                color = (0, 255, 0) if best_match_id != "Unknown" else (0, 0, 255)
                face_time = datetime.now().strftime("%I:%M %p")
                display_text = f"{best_match_id}: {best_match_name} ({face_time})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                recognized_faces.append(best_match_id)
            
            return frame, recognized_faces
        except Exception as e:
            logging.error(f"Error in face recognition: {e}")
            return frame, []

    def log_attendance(self, emp_id):
        """Log attendance in a JSON file and display attendance log."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = os.path.join(self.attendance_dir, f"Attendance_{date_str}.json")
            emp_name = self.stored_embeddings.get(emp_id, {}).get("name", "Unknown")
            time_str = datetime.now().strftime("%I:%M %p")
            face_time = datetime.now().strftime("%H:%M:%S")

            if os.path.exists(filename):
                with open(filename, "r") as file:
                    attendance_data = json.load(file)
            else:
                attendance_data = {}

            if emp_id in attendance_data:
                return False

            attendance_data[emp_id] = {"Name": emp_name, "Time": time_str}
            with open(filename, "w") as file:
                json.dump(attendance_data, file, indent=4)

            logging.info(f"Attendance logged: {emp_id} - {emp_name} at {time_str}")

            

            return True
        except Exception as e:
            logging.error(f"Error logging attendance: {e}")
            return False

    def show_actual_image(self, emp_id,emp_name):
        """Display the actual saved image of the recognized face."""
        image_path = os.path.join(self.image_dir, f"IWI-{emp_id}- {emp_name}", "face", "IWI-{emp_id}.jpg")
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            cv2.imshow("Recognized Face", img)


    def start_recognition(self):
        """Start face recognition and attendance logging."""
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, recognized_faces = self.recognize_faces(frame)
            cv2.imshow("Face Recognition", frame)
            
            for emp_id in recognized_faces:
                if emp_id not in self.recorded_today and emp_id != "Unknown":
                    if self.log_attendance(emp_id):
                        emp_name = self.stored_embeddings[emp_id]['name']
                        self.speak(f"Attendance taken for {emp_name}")
                        self.show_actual_image(emp_id,emp_name)
                        self.recorded_today.add(emp_id)
                        time.sleep(5)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognition()
    recognizer.start_recognition()
