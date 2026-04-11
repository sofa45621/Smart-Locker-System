import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import cv2
import os
import serial
import shutil
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

# ==========================================
# ⚙️ 1. 系統核心設定與硬體連線
# ==========================================
TOLERANCE = 0.35
base_dir = "facelook"

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

try:
    ser = serial.Serial('COM4', 115200, timeout=1)
    print("✅ ESP32 序列埠連線成功")
except Exception as serial_err:
    print(f"⚠️ 序列埠未連線，進入模擬模式. 錯誤: {serial_err}")
    ser = None

active_lockers = {}

print("⏳ 正在同步置物櫃資料...")
# 這裡把變數改名 (加上 init_) 以免跟後面的函數衝突
for folder_name in os.listdir(base_dir):
    if folder_name.isdigit():
        init_folder_path = os.path.join(base_dir, folder_name)
        init_photo_path = os.path.join(init_folder_path, "face.jpg")
        init_time_path = os.path.join(init_folder_path, "time.txt")

        if os.path.exists(init_photo_path):
            init_img = face_recognition.load_image_file(init_photo_path)
            init_encodings = face_recognition.face_encodings(init_img)
            if init_encodings:
                sync_start_time = time.time()
                if os.path.exists(init_time_path):
                    with open(init_time_path, "r") as f_read:
                        sync_start_time = float(f_read.read())

                active_lockers[int(folder_name)] = {
                    "encoding": init_encodings[0],
                    "start_time": sync_start_time
                }
print(f"📦 目前使用中的置物櫃數量: {len(active_lockers)} 個\n")

# ==========================================
# 🛠️ 2. 共用工具函數
# ==========================================
def get_next_id():
    existing_folders = [int(f) for f in os.listdir(base_dir) if f.isdigit()]
    return max(existing_folders) + 1 if existing_folders else 1

def trigger_door(locker_id):
    cmd = f"O{int(locker_id):02d}"
    if ser:
        try:
            ser.write(cmd.encode())
            print(f"⚡ 已發送開鎖指令 [{cmd}] 給 ESP32。")
        except Exception as send_err:
            print(f"⚠️ 序列埠傳送失敗: {send_err}")
    else:
        print(f"⚡ (模擬) 第 {locker_id} 號櫃已開啟，發送指令: {cmd}")

def decode_base64_image(b64_str):
    img_data = base64.b64decode(b64_str)
    numpy_arr = np.frombuffer(img_data, np.uint8)
    img_rgb = cv2.imdecode(numpy_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

# ==========================================
# 🌐 3. Web API 路由
# ==========================================
@app.route('/api/store', methods=['POST'])
def store_face():
    data = request.json
    face_img = decode_base64_image(data['image'])
    face_encodings = face_recognition.face_encodings(face_img)

    if len(face_encodings) == 1:
        current_encoding = face_encodings[0]

        if len(active_lockers) > 0:
            known_ids = list(active_lockers.keys())
            known_encodings = [v["encoding"] for v in active_lockers.values()]
            distances = face_recognition.face_distance(known_encodings, current_encoding)
            best_match_idx = np.argmin(distances)

            if distances[best_match_idx] < TOLERANCE:
                matched_id = known_ids[best_match_idx]
                return jsonify({"success": False, "msg": f"您已經在使用第 {matched_id} 號櫃！"})

        # 分配櫃子 (這段現在縮進正確了)
        new_id = get_next_id()
        user_folder = os.path.join(base_dir, str(new_id))
        os.makedirs(user_folder, exist_ok=True)
        cv2.imwrite(os.path.join(user_folder, "face.jpg"), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

        current_start_time = time.time()
        with open(os.path.join(user_folder, "time.txt"), "w") as f_write:
            f_write.write(str(current_start_time))

        active_lockers[new_id] = {
            "encoding": current_encoding,
            "start_time": current_start_time
        }

        print(f"\n✅ 註冊成功！您是第 {new_id} 號客人。")
        trigger_door(new_id)
        return jsonify({"success": True, "locker_id": new_id, "msg": "存物成功，已安全建檔"})

    return jsonify({"success": False, "msg": "畫面中找不到人臉或有多張人臉"})

@app.route('/api/retrieve', methods=['POST'])
def retrieve_face():
    data = request.json
    face_img = decode_base64_image(data['image'])
    face_encodings = face_recognition.face_encodings(face_img)

    if len(face_encodings) > 0:
        current_encoding = face_encodings[0]
        if len(active_lockers) > 0:
            known_ids = list(active_lockers.keys())
            known_encodings = [v["encoding"] for v in active_lockers.values()]
            distances = face_recognition.face_distance(known_encodings, current_encoding)

            best_match_idx = np.argmin(distances)
            best_dist = distances[best_match_idx]
            matched_id = known_ids[best_match_idx]

            print(f"\n📊 [AI 評分報告] 判定為 {matched_id} 號 | 距離分數: {best_dist:.3f}")

            if best_dist < TOLERANCE:
                print(f"🔓 解鎖成功！歡迎回來，第 {matched_id} 號客人。")
                trigger_door(matched_id)
                shutil.rmtree(os.path.join(base_dir, str(matched_id)), ignore_errors=True)
                del active_lockers[matched_id]
                return jsonify({"success": True, "locker_id": matched_id, "msg": "取物成功"})
            else:
                return jsonify({"success": False, "msg": f"驗證失敗，分數: {best_dist:.2f}"})
        return jsonify({"success": False, "msg": "無存物紀錄"})
    return jsonify({"success": False, "msg": "找不到人臉"})

@app.route('/api/status', methods=['GET'])
def get_status():
    status_report = []
    for i in range(1, 13): # 支援到 12 號櫃
        if i in active_lockers:
            duration_min = round((time.time() - active_lockers[i]["start_time"]) / 60, 1)
            status_report.append({
                "id": i, "occupied": True, "duration_min": duration_min
            })
        else:
            status_report.append({"id": i, "occupied": False})
    return jsonify(status_report)

@app.route('/api/admin/clear', methods=['POST'])
def admin_clear_locker():
    data = request.json
    l_id = int(data.get('locker_id'))
    pwd = data.get('password')
    if pwd == "888":
        if l_id in active_lockers:
            shutil.rmtree(os.path.join(base_dir, str(l_id)), ignore_errors=True)
            del active_lockers[l_id]
            trigger_door(l_id)
            return jsonify({"success": True, "msg": f"已清空 {l_id} 號"})
        return jsonify({"success": False, "msg": "無此紀錄"})
    return jsonify({"success": False, "msg": "密碼錯誤"})

if __name__ == '__main__':
    print("🚀 FaceLock 伺服器啟動中...")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)