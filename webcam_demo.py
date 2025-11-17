import cv2
import numpy as np
from bunny_vision_120 import BunnyVision120

def webcam_demo():
    print("🚀 Starting Real-Time Mask Detection Demo!")
    print("Press 'q' to quit")
    
    # Load trained model
    bunny = BunnyVision120(img_size=120)
    bunny.load_model('bunny_vision_120.h5')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Load face cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Resize to model input size
            face_resized = cv2.resize(face_img, (120, 120))
            
            # Predict mask
            pred_class, confidence, _ = bunny.predict_image(face_resized)
            
            # Draw bounding box and label
            if pred_class == 1:  # With mask
                color = (0, 255, 0)  # Green
                label = f"MASK ON ({confidence:.1f}%)"
            else:  # Without mask
                color = (0, 0, 255)  # Red
                label = f"NO MASK ({confidence:.1f}%)"
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show frame
        cv2.imshow('Bunny Vision 120 - Real-Time Mask Detection', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Demo ended!")

if __name__ == "__main__":
    webcam_demo()