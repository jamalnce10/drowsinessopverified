import os
import io
import time
import logging
from datetime import datetime
from scipy.spatial import distance
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import mediapipe as mp
import requests
import cv2
import threading
import wave

# -------------------------------
# Logger setup
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceEyeMonitorApp")

# -------------------------------
# Mediapipe Setup
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153]
RIGHT_EYE = [362, 382, 381, 380, 374, 390, 249, 263, 466, 388, 387, 386]
MOUTH_INNER = [78, 95, 88, 178, 181, 85, 84, 13, 311, 308, 324, 317, 14, 87, 86, 179]

CSV_FILE = "monitoring_logs.csv"

# -------------------------------
# Session State
# -------------------------------
if "log" not in st.session_state:
    st.session_state.log = []
if "last_telegram_alert" not in st.session_state:
    st.session_state.last_telegram_alert = 0
if "last_alert_frame" not in st.session_state:
    st.session_state.last_alert_frame = None
if "continuous_running" not in st.session_state:
    st.session_state.continuous_running = False

# -------------------------------
# Helper functions
# -------------------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[11])
    B = distance.euclidean(eye[2], eye[10])
    C = distance.euclidean(eye[3], eye[9])
    D = distance.euclidean(eye[4], eye[8])
    E = distance.euclidean(eye[5], eye[7])
    F = distance.euclidean(eye[0], eye[6])
    return (A + B + C + D + E) / (5.0 * F)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[15])
    B = distance.euclidean(mouth[2], mouth[14])
    C = distance.euclidean(mouth[3], mouth[13])
    D = distance.euclidean(mouth[4], mouth[12])
    E = distance.euclidean(mouth[5], mouth[11])
    F = distance.euclidean(mouth[0], mouth[6])
    return (A + B + C + D + E) / (5.0 * F)

def save_log(event_type, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {event_type}: {message}"
    st.session_state.log.insert(0, entry)
    new_entry = pd.DataFrame([[timestamp, event_type, message]], columns=["Timestamp", "Type", "Message"])
    if not os.path.exists(CSV_FILE):
        new_entry.to_csv(CSV_FILE, index=False)
    else:
        new_entry.to_csv(CSV_FILE, mode="a", header=False, index=False)

def send_telegram_alert(bot_token, chat_id, message, image_pil=None):
    if not bot_token or not chat_id:
        return False
    now = time.time()
    if now - st.session_state.last_telegram_alert < 7:
        return False
    base_url = f"https://api.telegram.org/bot{bot_token}"
    text_url = base_url + "/sendMessage"
    photo_url = base_url + "/sendPhoto"
    try:
        requests.post(text_url, data={"chat_id": chat_id, "text": message}, timeout=5)
        if image_pil:
            bio = io.BytesIO()
            image_pil.save(bio, format="JPEG")
            bio.seek(0)
            requests.post(photo_url, files={"photo": ("alert.jpg", bio, "image/jpeg")},
                          data={"chat_id": chat_id, "caption": message}, timeout=8)
        st.session_state.last_telegram_alert = now
        return True
    except Exception as e:
        save_log("Telegram", f"Failed to send alert: {e}")
        return False

# -------------------------------
# Generate in-memory beep WAV
# -------------------------------
def generate_beep_audio():
    sample_rate = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    signal_int16 = np.int16(signal * 32767)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal_int16.tobytes())
    buf.seek(0)
    return buf.read()

BEEP_AUDIO = generate_beep_audio()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🚨 Face & Eye Monitoring System")
st.markdown("### Snapshot & Continuous Drowsiness/Yawning Detection")

mode = st.sidebar.radio("Mode", ["Snapshot Mode", "Continuous Mode"])
EAR_THRESHOLD = st.sidebar.slider("Eye Aspect Ratio Threshold", 0.1, 0.5, 0.25)
MOUTH_THRESHOLD = st.sidebar.slider("Mouth Aspect Ratio Threshold", 0.3, 0.9, 0.6)

st.sidebar.markdown("### Telegram Alerts (optional)")
tele_token = st.sidebar.text_input("Bot Token", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
tele_chat_id = st.sidebar.text_input("Chat ID", value=os.getenv("TELEGRAM_CHAT_ID", ""))
tele_enable = st.sidebar.checkbox("Enable Telegram Alerts", value=False)

# -------------------------------
# Snapshot Mode
# -------------------------------
if mode == "Snapshot Mode":
    st.markdown("#### Take a snapshot for analysis")
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    camera_image = st.camera_input("Capture image")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        results = face_mesh.process(image_np)
        alarm_triggered = False
        status_message = "✅ All clear!"
        ear, mar = None, None
        draw = ImageDraw.Draw(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(lm.x*w), int(lm.y*h)) for lm in face_landmarks.landmark]
                left_eye = [landmarks[i] for i in LEFT_EYE]
                right_eye = [landmarks[i] for i in RIGHT_EYE]
                mouth = [landmarks[i] for i in MOUTH_INNER]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear)/2.0
                mar = mouth_aspect_ratio(mouth)
                if ear < EAR_THRESHOLD:
                    alarm_triggered = True
                    status_message = "😴 Drowsiness Detected!"
                if mar > MOUTH_THRESHOLD:
                    alarm_triggered = True
                    if status_message == "✅ All clear!":
                        status_message = ""
                    status_message += " 😮 Yawning!"
                if alarm_triggered:
                    draw.rectangle((0,0,w,h), outline="red", width=10)
                    draw.text((50,50), f"{status_message}\nEAR:{ear:.2f}\nMAR:{mar:.2f}", fill="red", font=ImageFont.load_default())
                    save_log("Webcam", f"{status_message} EAR:{ear:.2f} MAR:{mar:.2f}")
                else:
                    draw.rectangle((0,0,w,h), outline="green", width=10)
                    draw.text((50,50), f"{status_message}\nEAR:{ear:.2f}\nMAR:{mar:.2f}", fill="green", font=ImageFont.load_default())
        else:
            alarm_triggered = True
            status_message = "❌ No Face Detected!"
            draw.text((50,50), status_message, fill="red", font=ImageFont.load_default())
            save_log("Webcam", status_message)

        if alarm_triggered:
            st.audio(BEEP_AUDIO, format="audio/wav", autoplay=True)
            if tele_enable and tele_token and tele_chat_id:
                send_telegram_alert(tele_token, tele_chat_id,
                                    f"{status_message} EAR:{ear if ear else 0:.2f} MAR:{mar if mar else 0:.2f}", image)

        st.image(image, caption="Processed Image")

# -------------------------------
# Continuous Mode
# -------------------------------
elif mode == "Continuous Mode":
    st.sidebar.markdown("#### Controls")
    start_stop = st.sidebar.checkbox("Start Continuous Monitoring")

    video_placeholder = st.empty()
    status_text = st.empty()
    last_frame_download = st.empty()

    face_mesh_live = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)

    if start_stop:
        st.session_state.continuous_running = True
    else:
        st.session_state.continuous_running = False

    if st.session_state.continuous_running:
        ret, frame = st.session_state.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            results = face_mesh_live.process(frame_rgb)
            alarm_triggered = False
            status_message = "✅ All clear!"
            ear, mar = None, None

            if results.multi_face_landmarks:
                landmarks = [(int(lm.x*w), int(lm.y*h)) for lm in results.multi_face_landmarks[0].landmark]
                left_eye = [landmarks[i] for i in LEFT_EYE]
                right_eye = [landmarks[i] for i in RIGHT_EYE]
                mouth = [landmarks[i] for i in MOUTH_INNER]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear)/2.0
                mar = mouth_aspect_ratio(mouth)
                if ear < EAR_THRESHOLD:
                    alarm_triggered = True
                    status_message = "😴 Drowsiness Detected!"
                if mar > MOUTH_THRESHOLD:
                    alarm_triggered = True
                    if status_message == "✅ All clear!":
                        status_message = ""
                    status_message += " 😮 Yawning!"
                cv2.putText(frame, f"{status_message} EAR:{ear:.2f} MAR:{mar:.2f}", (30,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if alarm_triggered else (0,255,0), 2)
            else:
                alarm_triggered = True
                status_message = "❌ No Face Detected!"
                cv2.putText(frame, status_message, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

            if alarm_triggered:
                st.session_state.last_alert_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                threading.Thread(target=lambda: st.audio(BEEP_AUDIO, format="audio/wav", autoplay=True)).start()
                if tele_enable and tele_token and tele_chat_id:
                    img_pil = Image.fromarray(st.session_state.last_alert_frame)
                    send_telegram_alert(tele_token, tele_chat_id,
                                        f"{status_message} EAR:{ear if ear else 0:.2f} MAR:{mar if mar else 0:.2f}", img_pil)
                save_log("Continuous", f"{status_message} EAR:{ear if ear else 0:.2f} MAR:{mar if mar else 0:.2f}")

            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    else:
        if 'cap' in st.session_state:
            st.session_state.cap.release()
            del st.session_state.cap

    if st.session_state.get("last_alert_frame") is not None:
        pil_frame = Image.fromarray(st.session_state.last_alert_frame)
        buf = io.BytesIO()
        pil_frame.save(buf, format="JPEG")
        buf.seek(0)
        last_frame_download.download_button("📥 Download Last Captured Frame", buf, file_name="last_alert.jpg", mime="image/jpeg")

# -------------------------------
# Logs
# -------------------------------
st.markdown("---")
st.markdown("### Recent Logs")
log_placeholder = st.empty()
with log_placeholder.container():
    st.text_area("Logs", "\n".join(st.session_state.log[:10]), height=200, disabled=True)

