from email.mime.text import MIMEText
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from tinydb import TinyDB, Query
import os
import cv2
import face_recognition
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import shutil
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastdtw import fastdtw
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Set paths for FFmpeg
AudioSegment.converter = "C:/ProgramData/chocolatey/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/ProgramData/chocolatey/bin/ffprobe.exe"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_VIDEOS_DIR = "user_videos"
os.makedirs(USER_VIDEOS_DIR, exist_ok=True)

db = TinyDB("database.json")

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_ADDRESS = "snehal.22420227@viit.ac.in"  # Replace with your email
EMAIL_PASSWORD = "cqif pmdd lmlx fxzy"  # Replace with your email password

def send_email(to_email: str, subject: str, body: str, attachment_path: str):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach the photo
    with open(attachment_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(attachment_path)}",
        )
        msg.attach(part)

    # Send the email
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())

def save_video_file(username: str, video_file: UploadFile):
    file_path = os.path.join(USER_VIDEOS_DIR, f"{username}.webm")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
    return file_path

def extract_face_from_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            print(f"Face detected at: {face_locations}")
            video_capture.release()
            return face_recognition.face_encodings(frame, known_face_locations=[face_locations[0]])[0]
    video_capture.release()
    print("No face detected in video")
    return None

def compare_faces(registered_video_path, login_video_path):
    registered_face_encoding = extract_face_from_video(registered_video_path)
    login_face_encoding = extract_face_from_video(login_video_path)
    if registered_face_encoding is None or login_face_encoding is None:
        return False
    return bool(face_recognition.compare_faces([registered_face_encoding], login_face_encoding, tolerance=0.5)[0])

def extract_audio_from_video(video_path):
    try:
        audio = AudioSegment.from_file(video_path)
        wav_path = video_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")
        print(f"Extracted audio saved at: {wav_path}")
        return wav_path
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def compare_audio(registered_video_path, login_video_path):
    reg_wav_path = extract_audio_from_video(registered_video_path)
    login_wav_path = extract_audio_from_video(login_video_path)
    reg_audio_features = extract_audio_features(reg_wav_path)
    login_audio_features = extract_audio_features(login_wav_path)
    os.remove(reg_wav_path)
    os.remove(login_wav_path)
    return 1 - cosine(reg_audio_features, login_audio_features) > 0.5

def compare_key(stored_template, login_template):
    print(f"Stored Keystrokes: {stored_template}")
    print(f"Login Keystrokes: {login_template}")
    distance, _ = fastdtw(stored_template, login_template)
    print(f"Keystroke Distance: {distance}")
    return distance < 300.0

@app.post("/signup")
async def signup(username: str = Form(...), password: str = Form(...), video: UploadFile = File(...), keystroke_timings: str = Form(...)):
    if db.search(Query().uname == username):
        return JSONResponse(content={"success": False, "error": "Username already exists."}, status_code=400)
    video_file_path = save_video_file(username, video)
    keystroke_data = json.loads(keystroke_timings)
    db.insert({'uname': username, 'password': password, 'video_file': video_file_path, 'keystroke_timings': keystroke_data, 'failed_attempts': 0})
    return JSONResponse(content={"success": True, "message": "Signup successful"}, status_code=200)

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), video: UploadFile = File(...), keystroke_timings: str = Form(...), cursor_movements: str = Form(...)):
    result = db.search(Query().uname == username)
    if not result:
        return JSONResponse(content={"success": False, "error": "User does not exist."}, status_code=404)
    
    user = result[0]
    user_email = user['uname'] # Fetch user email from database

    # Save the login video temporarily
    login_video_path = os.path.join(USER_VIDEOS_DIR, f"temp_{username}.webm")
    with open(login_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Perform authentication checks
    registered_video_path = user['video_file']
    is_same_face = compare_faces(registered_video_path, login_video_path)
    is_same_voice = compare_audio(registered_video_path, login_video_path)
    stored_template = user.get("keystroke_timings", [])
    login_template = json.loads(keystroke_timings)
    is_same_keystroke = compare_key(stored_template, login_template)
    
    # Check if authentication failed
    if not (is_same_face and is_same_voice and is_same_keystroke):
        # Increment failed attempts
        failed_attempts = user.get('failed_attempts', 0) + 1
        db.update({'failed_attempts': failed_attempts}, Query().uname == username)
        print(f"User email: {user_email}")  # Debugging statement

        # Check if failed attempts >= 3
        if failed_attempts >= 3 and user_email:
            video_capture = cv2.VideoCapture(login_video_path)
            ret, frame = video_capture.read()
            video_capture.release()  # Release video capture object

            if ret:
                photo_path = os.path.join(USER_VIDEOS_DIR, f"temp_{username}.jpg")
                cv2.imwrite(photo_path, frame)

                # Send email with the photo
                send_email(
                    to_email=user_email,  # Fetch email from database
                    subject="⚠️ Security Alert: Failed Login Attempts!",
                    body=f"Someone has tried to access your account ({username}) 3 times unsuccessfully. See the attached photo.",
                    attachment_path=photo_path,
                )
                print("📧 Security alert email sent!")

                # ✅ Ensure `photo_path` exists before deleting it
                if os.path.exists(photo_path):
                    os.remove(photo_path)

        # ✅ Delete login video **AFTER** processing
        if os.path.exists(login_video_path):
            os.remove(login_video_path)

        return JSONResponse(content={
            "success": False,
            "error": "Authentication failed.",
            "face_match": bool(is_same_face),  # Convert NumPy bool to Python bool
            "voice_match": bool(is_same_voice),
            "keystroke_match": bool(is_same_keystroke),
            "cursor_mov": len(json.loads(cursor_movements)) > 3
        }, status_code=401)
    
    # ✅ Reset failed attempts on successful login
    db.update({'failed_attempts': 0}, Query().uname == username)
    
    # ✅ Save authentication results in the database
    cursor_movements_list = json.loads(cursor_movements)
    cursor_movement_detected = len(cursor_movements_list) > 3
    db.update({
        "auth_results": {
            "face_match": bool(is_same_face),
            "voice_match": bool(is_same_voice),
            "keystroke_match": bool(is_same_keystroke),
            "cursor_mov": cursor_movement_detected
        }
    }, Query().uname == username)

    # ✅ Delete login video **after successful login**
    if os.path.exists(login_video_path):
        os.remove(login_video_path)

    return JSONResponse(content={
        "success": True,
        "message": "User authenticated successfully",
        "face_match": bool(is_same_face),
        "voice_match": bool(is_same_voice),
        "keystroke_match": bool(is_same_keystroke),
        "cursor_mov": cursor_movement_detected
    }, status_code=200)


@app.get("/dashboard/{username}")
async def get_dashboard(username: str):
    result = db.search(Query().uname == username)
    if not result or "auth_results" not in result[0]:
        return JSONResponse(content={"error": "User data not found."}, status_code=404)
    
    return JSONResponse(content=result[0]["auth_results"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)