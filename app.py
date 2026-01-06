# app.py - Optimized for Mini PC (16GB RAM)
from flask import Flask, render_template, Response, jsonify, send_from_directory, request, send_file
from io import BytesIO
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import qrcode
import subprocess
import gc  # Garbage collector untuk memory management
import requests
import random
import string


app = Flask(__name__)

# Buat folder jika belum ada
os.makedirs("static/captured", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("hasilnari", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Inisialisasi MediaPipe Pose dengan setting ringan
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # 0 = lite model (lebih ringan)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Variabel global
current_shake_count = 0
prev_hip_x = None
recording = False
out = None
last_frame = None
user_data = {}
processing_status = {"status": "idle", "progress": 0}
FFMPEG_PATH = None
AUDIO_PATH = "static/assets/sound/soundprimaction.wav"

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

def check_ffmpeg_installed():
    """Cek FFmpeg dengan minimal logging"""
    global FFMPEG_PATH
    
    # Cek imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_path):
            FFMPEG_PATH = ffmpeg_path
            return True, ffmpeg_path
    except:
        pass
    
    # Cek system PATH
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            FFMPEG_PATH = 'ffmpeg'
            return True, 'ffmpeg'
    except:
        pass
    
    FFMPEG_PATH = None
    return False, None

def compress_video_opencv_light(input_path, output_path):
    """
    Kompresi ringan dengan OpenCV - Optimized untuk Mini PC
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Turunkan resolusi ke 720p untuk hemat RAM
        new_width = 720
        new_height = 1280
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        if not out_writer.isOpened():
            cap.release()
            return False
        
        # Process dengan JPEG quality 70 (lebih ringan)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Compress
            _, encoded = cv2.imencode('.jpg', resized, encode_param)
            compressed_frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            out_writer.write(compressed_frame)
            frame_count += 1
            
            # Clear memory setiap 60 frames
            if frame_count % 60 == 0:
                gc.collect()
        
        cap.release()
        out_writer.release()
        gc.collect()  # Clear memory
        
        return os.path.exists(output_path)
        
    except Exception as e:
        return False

def compress_video_ffmpeg_light(input_path, output_path):
    """
    Kompresi dengan FFmpeg - Optimized untuk Mini PC
    """
    global FFMPEG_PATH
    
    if not FFMPEG_PATH:
        return False
    
    try:
        # Setting ringan untuk Mini PC
        cmd = [
            FFMPEG_PATH,
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', '30',  # CRF 30 untuk kompresi lebih tinggi
            '-preset', 'veryfast',  # Very fast untuk CPU ringan
            '-vf', 'scale=720:1280',  # 720p untuk hemat RAM
            '-maxrate', '1M',  # 1Mbps max
            '-bufsize', '2M',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '64k',
            '-ar', '44100',
            '-movflags', '+faststart',
            '-threads', '2',  # Limit threads agar tidak overload
            '-y',
            output_path
        ]
        
        # Run dengan minimal output
        process = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,  # No stdout logging
            stderr=subprocess.DEVNULL,  # No stderr logging
            timeout=90
        )
        
        return process.returncode == 0 and os.path.exists(output_path)
        
    except:
        return False

def create_final_video_light(raw_video_path, output_path, frame_overlay_path, dream_key):
    """
    Buat video final dengan processing ringan untuk Mini PC
    Width & height lebih kecil: 720x1280 (hemat 50% memory)
    """
    global processing_status
    processing_status["status"] = "processing"
    processing_status["progress"] = 0
    
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        processing_status["status"] = "error"
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = int(10 * fps)
    
    # RESOLUSI LEBIH RENDAH untuk hemat RAM: 720p instead of 1080p
    width, height = 720, 1280
    
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    if not out_writer.isOpened():
        processing_status["status"] = "error"
        return False

    # Load overlay
    overlay_img = None
    if os.path.exists(frame_overlay_path):
        overlay_img = cv2.imread(frame_overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is not None:
            overlay_img = cv2.resize(overlay_img, (width, height), interpolation=cv2.INTER_AREA)

    # Load dream image
    DREAM_IMAGE_MAP = {
        "bebas_cicilan": "dream1.png",
        "rumah_impian": "dream2.png",
        "keliling_dunia": "dream3.png",
        "sukses_berbisnis": "dream4.png",
        "karir_melesat": "dream5.png",
        "ketemu_jodoh": "dream6.png",
        "keluarga_bahagia": "dream7.png"
    }

    dream_image_filename = DREAM_IMAGE_MAP.get(dream_key, "dream1.png")
    dream_image_path = os.path.join("static/assets/dreamimage", dream_image_filename)

    dream_img = None
    if os.path.exists(dream_image_path):
        dream_img = cv2.imread(dream_image_path, cv2.IMREAD_UNCHANGED)
        if dream_img is not None:
            target_width = int(width * 0.8)
            h, w = dream_img.shape[:2]
            scale = target_width / w
            new_h = int(h * scale)
            dream_img = cv2.resize(dream_img, (target_width, new_h), interpolation=cv2.INTER_AREA)

    frame_count = 0
    last_frame_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop to 9:16
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
        
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)

        # Add dream image
        if dream_img is not None:
            y_offset = int(height * 0.10)
            h_dream, w_dream = dream_img.shape[:2]
            if y_offset + h_dream > height:
                y_offset = height - h_dream
            x_offset = (width - w_dream) // 2

            if dream_img.shape[2] == 4:
                alpha = dream_img[:, :, 3] / 255.0
                for c in range(3):
                    resized[y_offset:y_offset+h_dream, x_offset:x_offset+w_dream, c] = \
                        resized[y_offset:y_offset+h_dream, x_offset:x_offset+w_dream, c] * (1 - alpha) + \
                        dream_img[:, :, c] * alpha
            else:
                resized[y_offset:y_offset+h_dream, x_offset:x_offset+w_dream] = dream_img

        # Add overlay
        if overlay_img is not None and overlay_img.shape[2] == 4:
            alpha = overlay_img[:, :, 3] / 255.0
            for c in range(3):
                resized[:, :, c] = resized[:, :, c] * (1 - alpha) + overlay_img[:, :, c] * alpha
        elif overlay_img is not None:
            resized = cv2.addWeighted(resized, 0.7, overlay_img, 0.3, 0)

        out_writer.write(resized)
        last_frame_img = resized.copy()
        frame_count += 1
        
        # Update progress
        if total_frames > 0:
            processing_status["progress"] = int((frame_count / total_frames) * 60)
        
        # Clear memory setiap 50 frames
        if frame_count % 50 == 0:
            gc.collect()

    cap.release()

    # Extend video jika kurang dari 10 detik
    if frame_count < target_frames and last_frame_img is not None:
        remaining = target_frames - frame_count
        for _ in range(remaining):
            out_writer.write(last_frame_img)
            frame_count += 1
            processing_status["progress"] = int((frame_count / target_frames) * 60)

    out_writer.release()
    gc.collect()

    # === 🎥 LANGKAH BARU: PROSES VIDEO + AUDIO SECARA EKSPLISIT ===
    raw_final = output_path  # Ini adalah file akhir yang diinginkan

    # Cek apakah FFmpeg tersedia
    ffmpeg_available, _ = check_ffmpeg_installed()
    temp_video_for_audio = temp_output  # Ini adalah hasil render dari OpenCV

    # Jika FFmpeg tersedia → kompres dulu, lalu tambah audio
    if ffmpeg_available:
        print("[INFO] FFmpeg ditemukan. Melakukan kompresi...")
        compression_success = compress_video_ffmpeg_light(temp_video_for_audio, raw_final)
        if not compression_success:
            print("[WARNING] Kompresi FFmpeg gagal. Gunakan video tanpa kompresi.")
            try:
                os.replace(temp_video_for_audio, raw_final)
                compression_success = True
            except Exception as e:
                print(f"[ERROR] Gagal fallback ke video mentah: {e}")
                compression_success = False
        else:
            # Hapus file sementara jika kompresi sukses
            if os.path.exists(temp_video_for_audio):
                os.remove(temp_video_for_audio)
    else:
        # Jika FFmpeg tidak ada → langsung gunakan video mentah dari OpenCV
        print("[INFO] FFmpeg tidak ditemukan. Melewati kompresi.")
        try:
            os.replace(temp_video_for_audio, raw_final)
            compression_success = True
        except Exception as e:
            print(f"[ERROR] Gagal pindahkan video mentah: {e}")
            compression_success = False

    # === 🔊 TAMBAHKAN AUDIO (HANYA JIKA VIDEO FINAL ADA) ===
    if compression_success and os.path.exists(raw_final):
        print(f"[DEBUG] Menambahkan audio ke: {raw_final}")
        audio_path = AUDIO_PATH
        print(f"[DEBUG] Lokasi audio: {audio_path}")
        print(f"[DEBUG] File audio ada: {os.path.exists(audio_path)}")

        if os.path.exists(audio_path):
            final_with_audio = raw_final.replace('.mp4', '_with_audio.mp4')
            if add_audio_to_video(raw_final, final_with_audio):
                if os.path.exists(final_with_audio):
                    os.replace(final_with_audio, raw_final)
                    print("✅ Audio berhasil ditambahkan ke video.")
                else:
                    print("⚠️ File output audio gagal dibuat.")
            else:
                print("⚠️ Fungsi add_audio_to_video() mengembalikan False.")
        else:
            print("❌ File audio TIDAK DITEMUKAN! Periksa path:", audio_path)
    else:
        print("❌ Video final tidak tersedia atau kompresi gagal. Audio dilewati.")

    # === SELESAI ===
    processing_status["status"] = "completed"
    processing_status["progress"] = 100
    gc.collect()

    return compression_success
    processing_status["progress"] = 100
    gc.collect()
    
    return compression_success

def process_video_async(raw_video, frame_overlay, impian, dream_key):
    """Background processing dengan unique code"""
    global user_data

    # Ambil data user
    nama = user_data.get('nama', 'user').replace('_', ' ')  # Kembalikan spasi untuk tampilan
    ktp6 = user_data.get('ktp6', '000000')[:6]

    # Generate unique code SEKARANG
    unique_code = generate_unique_code()
    
    # Simpan ke user_data (untuk ditampilkan di final_result.html)
    user_data['unique_code'] = unique_code

    # Buat nama file final sesuai format: Nama-6digitKTP-UniqueCode.mp4
    # Ganti spasi dengan underscore untuk nama file
    safe_nama = user_data.get('nama', 'user')  # nama sudah underscore dari form
    final_filename = f"{safe_nama}-{ktp6}-{unique_code}.mp4"
    final_video = os.path.join("hasilnari", final_filename)

    # Jalankan proses video
    success = create_final_video_light(
        raw_video_path=raw_video,
        output_path=final_video,
        frame_overlay_path=frame_overlay,
        dream_key=dream_key
    )
    
    if success:
        # Simpan nama file final ke user_data untuk referensi di template
        user_data['final_video_filename'] = final_filename

        # Buat thumbnail
        cap = cv2.VideoCapture(final_video)
        ret, thumb = cap.read()
        if ret:
            thumb = cv2.resize(thumb, (360, 640), interpolation=cv2.INTER_AREA)
            thumb_filename = f"{safe_nama}_{ktp6}_thumbnail.jpg"
            cv2.imwrite(os.path.join("results", thumb_filename), thumb, 
                       [cv2.IMWRITE_JPEG_QUALITY, 80])
        cap.release()

        # Hapus raw video
        if os.path.exists(raw_video):
            os.remove(raw_video)
        
        # 🔥 UPLOAD KE BACKEND LOKAL
        upload_success = upload_to_backend(final_video, ktp6, unique_code)
        if not upload_success:
            print("⚠️ Gagal upload ke backend, tapi file tetap disimpan lokal.")
    
    gc.collect()

def gen_frames():
    """Video feed dengan resolusi lebih rendah untuk hemat bandwidth"""
    global current_shake_count, prev_hip_x, recording, out, last_frame
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return

    # Set resolusi kamera lebih rendah untuk hemat resource
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)  # 20 FPS cukup

    frame_skip = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        if recording and out is not None:
            out.write(frame)

        # Skip frames untuk hemat CPU (process setiap 2 frame)
        frame_skip += 1
        if frame_skip % 2 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                hip = results.pose_landmarks.landmark[23]
                detect_shake(hip.x)

        # Encode dengan quality rendah untuk hemat bandwidth
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def generate_unique_code():
    # 5 karakter acak (huruf besar + angka), lalu akhiri dengan '7'
    chars = string.ascii_uppercase + string.digits
    prefix = ''.join(random.choices(chars, k=6))
    return prefix + '7'

def upload_to_backend(video_path, ktp6, unique_code):
    """
    Upload video ke backend lokal (localhost:4000)
    """
    upload_url = "http://localhost:4000/v1/files/offline"
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
            response = requests.post(upload_url, files=files, timeout=120)
        
        if response.status_code in (200, 201):
            print(f" Upload sukses: {video_path}")
            return True
        else:
            print(f" Upload gagal: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f" Error saat upload: {str(e)}")
        return False

def add_audio_to_video(video_path, output_path_with_audio):
    """
    Tambahkan 10 detik pertama dari soundprimariaction.wav ke video.
    Video input diasumsikan TANPA audio.
    """
    global FFMPEG_PATH
    if not FFMPEG_PATH:
        return False

    audio_path = AUDIO_PATH
    if not os.path.exists(audio_path):
        print(" Audio file not found:", audio_path)
        return False

    try:
        # Potong 10 detik audio & gabungkan ke video
        cmd = [
            FFMPEG_PATH,
            '-y',
            '-i', video_path,                  # input video (tanpa audio)
            '-ss', '0',                        # mulai dari detik 0
            '-t', '10',                        # ambil 10 detik
            '-i', audio_path,                  # input audio
            '-c:v', 'copy',                    # copy video stream (tidak re-encode)
            '-c:a', 'aac',                     # encode audio ke AAC
            '-shortest',                       # hentikan sesuai durasi terpendek (10 detik)
            '-map', '0:v:0',                   # ambil video dari input ke-0
            '-map', '1:a:0',                   # ambil audio dari input ke-1
            '-threads', '2',
            output_path_with_audio
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60
        )

        return result.returncode == 0 and os.path.exists(output_path_with_audio)

    except Exception as e:
        print(f" Gagal menambahkan audio: {e}")
        return False

@app.route('/generate-qr')
def generate_qr():
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
    global recording, out, user_data, processing_status
    data = request.get_json()
    nama = data.get('nama', 'user').replace(" ", "_")
    ktp6 = data.get('ktp6', '000000')[:6]
    impian = data.get('impian', 'MIMPI JADI POL')
    frame_choice = data.get('frame_choice', 'frame1.png')
    dream_key = data.get('dream_key', 'bebas_cicilan')

    user_data = {
        "nama": nama,
        "ktp6": ktp6,
        "impian": impian.upper(),
        "frame_choice": frame_choice,
        "dream_key": dream_key
    }
    
    processing_status = {"status": "idle", "progress": 0}

    raw_video = f"hasilnari/{nama}_{ktp6}_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    global recording, out, user_data
    
    if recording and out is not None:
        try:
            time.sleep(0.3)
            if out.isOpened():
                out.release()
        except:
            pass
        out = None

    recording = False

    if not user_data or 'nama' not in user_data:
        return jsonify({"status": "error", "message": "No user data"})

    nama = user_data['nama']
    ktp6 = user_data['ktp6']
    frame_choice = user_data['frame_choice']
    dream_key = user_data['dream_key']
    impian = user_data.get('impian', 'MIMPI JADI POL')

    raw_video = f"hasilnari/{nama}_{ktp6}_raw.mp4"
    frame_overlay = f"static/assets/{frame_choice}"

    if not os.path.exists(raw_video):
        return jsonify({"status": "error", "message": "Raw video not found"})

    # Jalankan thread dengan parameter sesuai fungsi baru
    thread = threading.Thread(
        target=process_video_async,
        args=(raw_video, frame_overlay, impian, dream_key)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"status": "success", "message": "Processing started"})

@app.route('/check_processing')
def check_processing():
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

@app.route('/download')
def download():
    return render_template('download.html')

@app.route('/bgm-player')
def bgm_player():
    return render_template('bgm_player.html')

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Flask App...")
    ffmpeg_ok, ffmpeg_path = check_ffmpeg_installed()
    print(" FFmpeg tersedia:", ffmpeg_ok)
    if ffmpeg_ok:
        print("   Path:", ffmpeg_path)
    print(" File audio:", AUDIO_PATH)
    print("   Ada?", os.path.exists(AUDIO_PATH))
    print("=" * 50)
    app.run(debug=True, threaded=True, host='127.0.0.1', port=5000)