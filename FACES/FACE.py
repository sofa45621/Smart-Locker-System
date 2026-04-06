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
# ⚙️ 1. 系統核心設定與硬體連線 (完全繼承你的設定)
# ==========================================
TOLERANCE = 0.35  # 臉部辨識容錯率
base_dir = "facelook"

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

try:
    # 這裡保留你原本連線 STM32 的設定
    ser = serial.Serial('COM3', 115200, timeout=1)
    print("✅ STM32 序列埠連線成功")
except Exception as e:
    print(f"⚠️ 序列埠未連線，進入模擬模式. 錯誤: {e}")
    ser = None

active_lockers = {}

print("⏳ 正在同步置物櫃資料...")
for folder_name in os.listdir(base_dir):
    if folder_name.isdigit():
        photo_path = os.path.join(base_dir, folder_name, "face.jpg")
        if os.path.exists(photo_path):
            img = face_recognition.load_image_file(photo_path)
            encs = face_recognition.face_encodings(img)
            if encs:
                active_lockers[int(folder_name)] = encs[0]
print(f"📦 目前使用中的置物櫃數量: {len(active_lockers)} 個\n")


# ==========================================
# 🛠️ 2. 共用工具函數
# ==========================================
def get_next_id():
    existing_folders = [int(f) for f in os.listdir(base_dir) if f.isdigit()]
    return max(existing_folders) + 1 if existing_folders else 1


def trigger_door(locker_id):
    # 完全使用你寫的通訊協定 O01, O02...
    cmd = f"O{int(locker_id):02d}"
    if ser:
        try:
            ser.write(cmd.encode())
            print(f"⚡ 已發送開鎖指令 [{cmd}] 給 STM32。")
        except Exception as e:
            print(f"⚠️ 序列埠傳送失敗: {e}")
    else:
        print(f"⚡ (模擬) 第 {locker_id} 號櫃已開啟，發送指令: {cmd}")


def decode_base64_image(b64_str):
    """將網頁傳來的照片轉換為 OpenCV 格式"""
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ==========================================
# 🌐 3. Web API 路由 (取代原本的鍵盤監聽)
# ==========================================
@app.route('/api/store', methods=['POST'])
def store_face():
    data = request.json
    img = decode_base64_image(data['image'])
    face_encodings = face_recognition.face_encodings(img)

    if len(face_encodings) == 1:
        face_encoding = face_encodings[0]

        # 檢查是否已經註冊過 (你的拒絕重複註冊邏輯)
        if len(active_lockers) > 0:
            known_ids = list(active_lockers.keys())
            known_encodings = list(active_lockers.values())
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < TOLERANCE:
                matched_id = known_ids[best_match_index]
                print(f"\n🚫 註冊拒絕：您已經在使用第 {matched_id} 號櫃！")
                return jsonify({"success": False, "msg": f"您已經在使用第 {matched_id} 號櫃！"})

        # 存檔並分配櫃子
        new_id = get_next_id()
        user_folder = os.path.join(base_dir, str(new_id))
        os.makedirs(user_folder, exist_ok=True)
        # 儲存照片到本機
        cv2.imwrite(os.path.join(user_folder, "face.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        active_lockers[new_id] = face_encoding

        print(f"\n✅ 註冊成功！您是第 {new_id} 號客人。")
        trigger_door(new_id)
        return jsonify({"success": True, "locker_id": new_id, "msg": "存物成功，已安全建檔"})

    return jsonify({"success": False, "msg": "畫面中找不到人臉或有多張人臉"})


@app.route('/api/retrieve', methods=['POST'])
def retrieve_face():
    data = request.json
    img = decode_base64_image(data['image'])
    face_encodings = face_recognition.face_encodings(img)

    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        if len(active_lockers) > 0:
            known_ids = list(active_lockers.keys())
            known_encodings = list(active_lockers.values())
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            matched_id = known_ids[best_match_index]

            print("\n" + "=" * 40)
            print(f"📊 [AI 評分報告] 判定為 {matched_id} 號 | 距離分數: {best_distance:.3f}")
            print("=" * 40)

            if best_distance < TOLERANCE:
                print(f"🔓 解鎖成功！歡迎回來，第 {matched_id} 號客人。")
                trigger_door(matched_id)
                # 刪除資料夾與紀錄 (隱私保護)
                shutil.rmtree(os.path.join(base_dir, str(matched_id)), ignore_errors=True)
                del active_lockers[matched_id]

                return jsonify({"success": True, "locker_id": matched_id, "msg": f"取物成功，相似度達標。資料已銷毀"})
            else:
                return jsonify({"success": False, "msg": f"解鎖失敗！未達解鎖標準 (距離分數:{best_distance:.2f})"})
        return jsonify({"success": False, "msg": "目前沒有任何存物紀錄"})
    return jsonify({"success": False, "msg": "畫面中找不到人臉"})


if __name__ == '__main__':
    print("🚀 FaceLock 伺服器啟動中... 等待網頁連線...")
    app.run(host='127.0.0.1', port=5000, debug=True)