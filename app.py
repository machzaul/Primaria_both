# app.py
from flask import Flask, render_template, Response, jsonify, send_from_directory, request, redirect, url_for, send_file
from io import BytesIO
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import threading
import qrcode

app = Flask(__name__)

# Buat folder jika belum ada
os.makedirs("static/captured", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("hasilnari", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Inisialisasi MediaPipe Pose (hanya untuk deteksi, tidak digambar)
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
processing_status = {"status": "idle", "progress": 0}  # Status processing

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

def create_final_video(raw_video_path, output_path, frame_overlay_path, dream_key, width=1080, height=1920):
    """
    Buat video final 9:16 dengan frame overlay dan gambar dream.
    """
    global processing_status
    processing_status["status"] = "processing"
    processing_status["progress"] = 0
    
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka video mentah.")
        processing_status["status"] = "error"
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Target durasi 10 detik
    target_frames = int(10 * fps)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Muat frame overlay
    overlay_img = None
    if os.path.exists(frame_overlay_path):
        overlay_img = cv2.imread(frame_overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is not None:
            overlay_img = cv2.resize(overlay_img, (width, height), interpolation=cv2.INTER_AREA)

    # Mapping dream_key ke nama file gambar
    DREAM_IMAGE_MAP = {
        "bebas_cicilan": "dream1.png",
        "rumah_impian": "dream2.png",
        "keliling_dunia": "dream3.png",
        "sukses_berbisnis": "dream4.png",
        "karir_melesat": "dream5.png",
        "ketemu_jodoh": "dream6.png",
        "keluarga_bahagia": "dream7.png"
    }

    # Ambil nama file gambar berdasarkan dream_key
    dream_image_filename = DREAM_IMAGE_MAP.get(dream_key, "dream1.png")  # default
    dream_image_path = os.path.join("static/assets/dreamimage", dream_image_filename)

    # Muat gambar dream
    dream_img = None
    if os.path.exists(dream_image_path):
        dream_img = cv2.imread(dream_image_path, cv2.IMREAD_UNCHANGED)
        if dream_img is not None:
            # Resize agar lebar 80% dari width video (lebih kecil)
            target_width = int(width * 0.8)  # 80% dari 1080 = 864px
            h, w = dream_img.shape[:2]
            scale = target_width / w
            new_h = int(h * scale)
            dream_img = cv2.resize(dream_img, (target_width, new_h), interpolation=cv2.INTER_AREA)

    frame_count = 0
    frames_list = []  # Simpan frame jika perlu looping

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

        # Tambahkan gambar dream di atas video
        if dream_img is not None:
            # Tentukan posisi Y: 10% dari atas
            y_offset = int(height * 0.10)  # 10% dari atas
            h_dream, w_dream = dream_img.shape[:2]

            # Pastikan tidak melebihi batas frame
            if y_offset + h_dream > height:
                y_offset = height - h_dream

            # Tentukan posisi X: center horizontal
            x_offset = (width - w_dream) // 2

            # Overlay gambar dengan alpha jika ada
            if dream_img.shape[2] == 4:  # PNG dengan alpha
                alpha = dream_img[:, :, 3] / 255.0
                for c in range(3):
                    resized[y_offset:y_offset+h_dream, x_offset:x_offset+w_dream, c] = \
                        resized[y_offset:y_offset+h_dream, x_offset:x_offset+w_dream, c] * (1 - alpha) + \
                        dream_img[:, :, c] * alpha
            else:
                # Tanpa alpha — replace langsung
                resized[y_offset:y_offset+h_dream, x_offset:x_offset+w_dream] = dream_img

        # Tambahkan frame overlay
        if overlay_img is not None and overlay_img.shape[2] == 4:
            alpha = overlay_img[:, :, 3] / 255.0
            for c in range(3):
                resized[:, :, c] = resized[:, :, c] * (1 - alpha) + overlay_img[:, :, c] * alpha
        elif overlay_img is not None:
            resized = cv2.addWeighted(resized, 0.7, overlay_img, 0.3, 0)

        out_writer.write(resized)
        frames_list.append(resized.copy())  # Simpan frame untuk looping jika perlu
        frame_count += 1
        
        # Update progress
        if total_frames > 0:
            processing_status["progress"] = int((frame_count / total_frames) * 100)

    cap.release()

    # Jika durasi < 10 detik, loop frame terakhir sampai 10 detik
    if frame_count < target_frames:
        remaining = target_frames - frame_count
        last_frame = frames_list[-1] if frames_list else None
        for _ in range(remaining):
            out_writer.write(last_frame)
            frame_count += 1
            processing_status["progress"] = int((frame_count / target_frames) * 100)

    # Jika durasi > 10 detik, stop setelah 10 detik
    elif frame_count > target_frames:
        print(f"Video dipotong dari {frame_count} frame menjadi {target_frames} frame (10 detik).")

    out_writer.release()
    print(f"Video final dibuat: {frame_count} frame → {output_path}")
    processing_status["status"] = "completed"
    processing_status["progress"] = 100
    return True

def process_video_async(raw_video, final_video, frame_overlay, impian, dream_key):
    """Function untuk processing video di background thread"""
    success = create_final_video(
        raw_video_path=raw_video,
        output_path=final_video,
        frame_overlay_path=frame_overlay,
        dream_key=dream_key  # <-- kirim dream_key
    )
    
    if success:
        # Buat thumbnail
        nama = user_data.get('nama', 'user')
        ktp6 = user_data.get('ktp6', '000000')
        cap = cv2.VideoCapture(final_video)
        success, thumb = cap.read()
        if success:
            cv2.imwrite(f"results/{nama}_{ktp6}_thumbnail.jpg", thumb)
        cap.release()

        # Hapus raw video
        if os.path.exists(raw_video):
            os.remove(raw_video)

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

        # Hanya proses pose untuk deteksi shake, TIDAK GAMBAR LANDMARKS
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            hip = results.pose_landmarks.landmark[23]
            detect_shake(hip.x)

            # Tampilkan counter shake saja (opsional, bisa dihapus juga)


        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# === BARU: Generate QR Code dinamis ===
@app.route('/generate-qr')
def generate_qr():
    from io import BytesIO
    import qrcode
    url = request.args.get('url')
    if not url:
        return "URL required", 400

    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

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
    global recording, out, video_filename, user_data, processing_status
    data = request.get_json()
    nama = data.get('nama', 'user').replace(" ", "_")
    ktp6 = data.get('ktp6', '000000')[:6]
    impian = data.get('impian', 'MIMPI JADI POL')
    frame_choice = data.get('frame_choice', 'frame1.png')
    dream_key = data.get('dream_key', 'bebas_cicilan')  # <-- tambahkan ini

    user_data = {
        "nama": nama,
        "ktp6": ktp6,
        "impian": impian.upper(),
        "frame_choice": frame_choice,
        "dream_key": dream_key  # <-- simpan key dream
    }
    
    # Reset processing status
    processing_status = {"status": "idle", "progress": 0}

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
    dream_key = user_data['dream_key']  # <-- ambil dream_key

    raw_video = f"hasilnari/{nama}_{ktp6}_raw.mp4"
    final_video = f"hasilnari/{nama}_{ktp6}_final.mp4"
    frame_overlay = f"static/assets/{frame_choice}"

    if not os.path.exists(raw_video):
        return jsonify({"status": "error", "message": "Raw video not found"})

    # Mulai processing di background thread
    thread = threading.Thread(
        target=process_video_async,
        args=(raw_video, final_video, frame_overlay, impian, dream_key)  # <-- kirim dream_key
    )
    thread.daemon = True
    thread.start()

    return jsonify({"status": "success", "message": "Processing started"})

@app.route('/check_processing')
def check_processing():
    """Endpoint untuk mengecek status processing"""
    global processing_status
    return jsonify(processing_status)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('static/assets', filename)

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)

@app.route('/form')
def form():
    global processing_status
    # Reset processing status saat kembali ke form
    processing_status = {"status": "idle", "progress": 0}
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
    global user_data, processing_status
    
    # Reset processing status agar tidak loop
    processing_status = {"status": "idle", "progress": 0}
    
    if not user_data:
        user_data = {
            "nama": "User",
            "ktp6": "000000",
            "impian": "MIMPI JADI POL",
            "frame_choice": "frame1.png",
            "dream_key": "bebas_cicilan"
        }
    return render_template('final_result.html', user=user_data)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)