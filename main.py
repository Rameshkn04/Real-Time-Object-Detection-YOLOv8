import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model

# ===============================================================
# 🧠 STEP 1: DEEP LEARNING - CNN Verifier
# ===============================================================

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # 2 outputs: Verified / Uncertain
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Try to load trained CNN or create a new one
try:
    cnn_model = load_model("cnn_verifier.h5")
    print("✅ Loaded existing CNN model.")
except:
    cnn_model = build_cnn_model()
    print("⚙️ New CNN model created (untrained).")

# ===============================================================
# ⚡ STEP 2: YOLOv8 Object Detection
# ===============================================================
yolo_model = YOLO('yolov8l.pt')
print("✅ YOLOv8 model loaded.")

# ===============================================================
# 🧩 STEP 3: Reinforcement Learning Setup
# ===============================================================
confidence_threshold = 0.5   # start point
learning_rate = 0.05         # RL learning rate
reward_memory = []           # store recent feedback

def adjust_confidence(reward):
    """Reinforcement Learning function to adjust confidence dynamically."""
    global confidence_threshold
    confidence_threshold += learning_rate * reward
    confidence_threshold = np.clip(confidence_threshold, 0.3, 0.9)
    return confidence_threshold

# ===============================================================
# 📸 STEP 4: Capture and Process Video
# ===============================================================
cap = cv2.VideoCapture(0)

print("🚀 Starting Real-Time Detection with DL + RL...")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection
    results = yolo_model(frame)
    annotated_frame = results[0].plot()

    # Loop through detected boxes
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Skip low-confidence detections (dynamic threshold via RL)
        if conf < confidence_threshold:
            continue

        # Crop detected object
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Prepare for CNN verifier
        resized = cv2.resize(cropped, (64, 64))
        resized = np.expand_dims(resized, axis=0) / 255.0

        # Deep Learning (CNN) verification
        preds = cnn_model.predict(resized, verbose=0)
        label = np.argmax(preds)
        verified = True  # temporary testing mode

        # ==============================
        # 🤖 Reinforcement Learning Logic
        # ==============================
        if verified and conf > confidence_threshold:
            reward = +1   # good detection
        else:
            reward = -1   # wrong detection or low confidence

        # Update threshold adaptively
        confidence_threshold = adjust_confidence(reward)
        reward_memory.append(reward)
        if len(reward_memory) > 50:
            reward_memory.pop(0)

        # Display feedback info
        status_text = f"Verified:{verified} | Conf:{conf:.2f} | Thresh:{confidence_threshold:.2f}"
        color = (0, 255, 0) if verified else (0, 0, 255)
        cv2.putText(annotated_frame, status_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv8 + Deep Learning + RL", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Stream ended.")
