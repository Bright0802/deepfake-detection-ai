import cv2
import os

# Create dataset folder if it doesn't exist
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        count += 1

        file_path = os.path.join(dataset_path, f"face_{count}.jpg")
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Dataset Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 200:
        break

cap.release()
cv2.destroyAllWindows()