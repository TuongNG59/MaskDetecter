import cv2
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def main():
    # 1. Tải mô hình đã huấn luyện
    print("[INFO] Đang tải mô hình phân loại khẩu trang...")
    model_path = "mask_detector.h5"
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy file mô hình '{model_path}'. Vui lòng kiểm tra lại!")
        return
    
    mask_model = load_model(model_path)

    # 2. Kiểm tra và nạp file bộ phát hiện khuôn mặt Haar Cascade
    xml_filename = "haarcascade_frontalface_default.xml"
    xml_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

    if not os.path.exists(xml_filename):
        print(f"[INFO] Đang tải file {xml_filename}...")
        urllib.request.urlretrieve(xml_url, xml_filename)

    print("[INFO] Đang tải bộ phát hiện khuôn mặt...")
    face_cascade = cv2.CascadeClassifier(xml_filename)

    # 3. Kích hoạt Webcam
    print("[INFO] Đang khởi động Webcam...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[LỖI] Không thể kết nối với Webcam.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_roi = cv2.resize(face_roi, (224, 224))
            face_roi = img_to_array(face_roi)
            face_roi = preprocess_input(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)

            (mask, withoutMask) = mask_model.predict(face_roi, verbose=0)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label_text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Face Mask Detector - Press 'q' to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()