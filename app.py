# app.py
from flask import Flask, render_template, Response, jsonify, send_from_directory, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json

app = Flask(__name__)

# Buat folder jika belum ada
os.makedirs("static/captured", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("hasilnari", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variabel global
current_shake_count = 0
prev_hip_x = None
recording = False
out = None
video_filename = ""
last_frame = None
user_data = {}

def detect_shake(hip_x, threshold=0.02):
    global prev_hip_x, current_shake_count
    if prev_hip_x is None:
        prev_hip_x = hip_x
        return False

    diff = abs(hip_x - prev_hip_x)
    if diff > threshold:
        current_shake_count += 1
        prev_hip_x = hip_x
        return True
    prev_hip_x = hip_x
    return False

def create_final_video(raw_video_path, output_path, frame_overlay_path, impian_text, width=1080, height=1920):
    """
    Buat video final 9:16 dengan frame overlay dan teks impian.
    """
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka video mentah.")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Muat frame overlay
    overlay_img = None
    if os.path.exists(frame_overlay_path):
        overlay_img = cv2.imread(frame_overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is not None:
            overlay_img = cv2.resize(overlay_img, (width, height), interpolation=cv2.INTER_AREA)

    # Split teks
    text_lines = impian_text.split('\n')
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.5
    thickness = 3

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to 9:16
        h, w = frame.shape[:2]
        if w / h > 9 / 16:
            new_w = int(h * 9 / 16)
            left = (w - new_w) // 2
            cropped = frame[:, left:left+new_w]
        else:
            new_h = int(w * 16 / 9)
            if new_h > h:
                new_h = h
                new_w = int(h * 9 / 16)
                left = (w - new_w) // 2
                cropped = frame[:, left:left+new_w]
            else:
                top = (h - new_h) // 2
                cropped = frame[top:top+new_h, :]
        resized = cv2.resize(cropped, (width, height))

        # Tambahkan teks
        for i, line in enumerate(text_lines):
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = int(height * 0.08) + i * (text_size[1] + 10)
            cv2.putText(resized, line, (text_x, text_y),
                        font, font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)  # outline
            cv2.putText(resized, line, (text_x, text_y),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)  # fill

        # Tambahkan frame overlay
        if overlay_img is not None and overlay_img.shape[2] == 4:
            alpha = overlay_img[:, :, 3] / 255.0
            for c in range(3):
                resized[:, :, c] = resized[:, :, c] * (1 - alpha) + overlay_img[:, :, c] * alpha
        elif overlay_img is not None:
            resized = cv2.addWeighted(resized, 0.7, overlay_img, 0.3, 0)

        out_writer.write(resized)
        frame_count += 1

    cap.release()
    out_writer.release()
    print(f"Video final dibuat: {frame_count} frame → {output_path}")
    return True

def gen_frames():
    global current_shake_count, prev_hip_x, recording, out, last_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        if recording and out is not None:
            out.write(frame)

        last_frame = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            hip = results.pose_landmarks.landmark[23]
            detect_shake(hip.x)

            cv2.putText(frame, f"Shake: {current_shake_count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hasilnari/<path:filename>')
def serve_hasilnari(filename):
    return send_from_directory('hasilnari', filename)

@app.route('/get_shake_count')
def get_shake_count():
    global current_shake_count
    return jsonify({"shake_count": current_shake_count})

@app.route('/reset_shake_count')
def reset_shake_count():
    global current_shake_count, prev_hip_x
    current_shake_count = 0
    prev_hip_x = None
    return jsonify({"status": "success"})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, out, video_filename, user_data
    data = request.get_json()
    nama = data.get('nama', 'user').replace(" ", "_")
    ktp6 = data.get('ktp6', '000000')[:6]
    impian = data.get('impian', 'MIMPI JADI POL')
    frame_choice = data.get('frame_choice', 'frame1.png')

    user_data = {
        "nama": nama,
        "ktp6": ktp6,
        "impian": impian.upper(),
        "frame_choice": frame_choice
    }

    raw_video = f"hasilnari/{nama}_{ktp6}_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"status": "error", "message": "Camera not available"}), 500

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    out = cv2.VideoWriter(raw_video, fourcc, fps, (width, height))
    cap.release()

    recording = True
    return jsonify({"status": "success", "filename": raw_video})

@app.route('/stop_recording')
def stop_recording():
    global recording, out, last_frame, user_data
    if recording and out is not None:
        try:
            time.sleep(0.3)
            if out.isOpened():
                out.release()
        except Exception as e:
            print(f"Error releasing raw video: {e}")
        out = None

    recording = False

    if not user_data or 'nama' not in user_data:
        return jsonify({"status": "error", "message": "No user data"})

    nama = user_data['nama']
    ktp6 = user_data['ktp6']
    impian = user_data['impian']
    frame_choice = user_data['frame_choice']

    raw_video = f"hasilnari/{nama}_{ktp6}_raw.mp4"
    final_video = f"hasilnari/{nama}_{ktp6}_final.mp4"
    frame_overlay = f"static/assets/{frame_choice}"

    if not os.path.exists(raw_video):
        return jsonify({"status": "error", "message": "Raw video not found"})

    success = create_final_video(
        raw_video_path=raw_video,
        output_path=final_video,
        frame_overlay_path=frame_overlay,
        impian_text=impian.replace(' ', '\n')
    )

    if success:
        cap = cv2.VideoCapture(final_video)
        success, thumb = cap.read()
        if success:
            cv2.imwrite(f"results/{nama}_{ktp6}_thumbnail.jpg", thumb)
        cap.release()

        if os.path.exists(raw_video):
            os.remove(raw_video)

        return jsonify({"status": "success", "final_video": final_video})
    else:
        return jsonify({"status": "error", "message": "Gagal membuat video final"})

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('static/assets', filename)

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/select_frame')
def select_frame():
    return render_template('select_frame.html')

@app.route('/dream_selection')
def dream_selection():
    return render_template('dream_selection.html')

@app.route('/dance_prep')
def dance_prep():
    return render_template('dance_prep.html')

@app.route('/loading_result')
def loading_result():
    return render_template('loading_result.html')

@app.route('/final_result')
def final_result():
    global user_data
    if not user_data:
        user_data = {
            "nama": "User",
            "ktp6": "000000",
            "impian": "MIMPI JADI POL",
            "frame_choice": "frame1.png"
        }
    return render_template('final_result.html', user=user_data)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)