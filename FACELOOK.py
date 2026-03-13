import cv2
import face_recognition
import os
import serial
import shutil
import numpy as np
import time

# ==========================================
# ⚙️ 0. 系統核心參數設定 (展場調校專區)
# ==========================================
TOLERANCE = 0.40  # 臉部辨識容錯率 (<0.40 為及格)
TIMEOUT_SECONDS = 30  # 閒置超時退回主畫面秒數
ADMIN_PWD = "888"  # 管理員超級密碼
EAR_THRESHOLD = 0.22  # 眨眼偵測門檻 (數值越小代表眼睛閉得越緊)

# ==========================================
# 📁 1. 初始化與環境設定
# ==========================================
base_dir = "facelook"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

try:
    ser = serial.Serial('COM3', 115200, timeout=1)
    print("✅ STM32 序列埠連線成功")
except Exception as e:
    print(f"⚠️ 序列埠未連線，進入無硬體模擬模式. 錯誤: {e}")
    ser = None

video_url = "http://192.168.0.150:4747/video"
video_capture = cv2.VideoCapture(video_url)

# ==========================================
# 📊 2. 系統狀態與動態資料庫
# ==========================================
STATE_MAIN_MENU = 0
STATE_CONFIRM_STORE = 1
STATE_CONFIRM_OPEN = 2

current_state = STATE_MAIN_MENU
state_start_time = 0  # 紀錄進入狀態的時間 (用於超時防呆)
has_blinked = False  # 紀錄是否通過活體眨眼測試

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


# --- 數學公式：計算眼睛長寬比 (EAR) ---
def calculate_ear(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0


def get_next_id():
    return max(active_lockers.keys()) + 1 if active_lockers else 1


# --- 升級：多櫃位硬體通訊協定 ---
def trigger_door(locker_id):
    cmd = f"O{int(locker_id):02d}"  # 組合指令 (例如: O01, O02)
    if ser:
        try:
            ser.write(cmd.encode())
            print(f"⚡ 已發送開鎖指令 [{cmd}] 給 STM32。")
        except Exception as e:
            print(f"⚠️ 序列埠傳送失敗: {e}")
    else:
        print(f"⚡ (模擬) 第 {locker_id} 號櫃已開啟，發送指令: {cmd}")


# ==========================================
# 📥 3. 存物邏輯
# ==========================================
def process_store(frame, face_encodings):
    global current_state
    if len(face_encodings) == 1:
        face_encoding = face_encodings[0]

        if len(active_lockers) > 0:
            known_ids = list(active_lockers.keys())
            known_encodings = list(active_lockers.values())
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < TOLERANCE:
                matched_id = known_ids[best_match_index]
                print(f"\n🚫 註冊拒絕：您已經在使用第 {matched_id} 號櫃！")
                current_state = STATE_MAIN_MENU
                return

        new_id = get_next_id()
        user_folder = os.path.join(base_dir, str(new_id))
        os.makedirs(user_folder, exist_ok=True)
        cv2.imwrite(os.path.join(user_folder, "face.jpg"), frame)
        active_lockers[new_id] = face_encoding

        print(f"\n✅ 註冊成功！您是第 {new_id} 號客人。")
        trigger_door(new_id)
    else:
        print("\n⚠️ 畫面中找不到人臉或有多張人臉，註冊失敗。")

    current_state = STATE_MAIN_MENU


# ==========================================
# 📤 4. 取物邏輯
# ==========================================
def process_retrieve(face_encodings):
    global current_state
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        known_ids = list(active_lockers.keys())
        known_encodings = list(active_lockers.values())

        if len(known_encodings) > 0:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            matched_id = known_ids[best_match_index]

            print("\n" + "=" * 40)
            print(f"📊 [AI 評分報告] 判定為 {matched_id} 號客人 | 距離分數: {best_distance:.3f}")
            print("=" * 40)

            if best_distance < TOLERANCE:
                print(f"🔓 解鎖成功！歡迎回來，第 {matched_id} 號客人。")
                trigger_door(matched_id)

                shutil.rmtree(os.path.join(base_dir, str(matched_id)), ignore_errors=True)
                del active_lockers[matched_id]
                print(f"🗑️ 隱私保護啟動：第 {matched_id} 號客人的資料已銷毀。")
            else:
                print("❌ 解鎖失敗！未達解鎖標準，系統判定為陌生人防護中。")
        else:
            print("\n⚠️ 系統發生錯誤，找不到使用紀錄。")
    else:
        print("\n⚠️ 畫面中找不到人臉，取物失敗。")

    current_state = STATE_MAIN_MENU


# ==========================================
# 🖥️ 5. 影像主迴圈與介面
# ==========================================
print("🎥 智慧置物櫃 Kiosk 系統啟動！")

while True:
    ret, frame = video_capture.read()
    if not ret: break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # --- 閒置超時防呆機制 ---
    if current_state != STATE_MAIN_MENU:
        if time.time() - state_start_time > TIMEOUT_SECONDS:
            print(f"\n⏳ 閒置超過 {TIMEOUT_SECONDS} 秒，系統自動返回主選單以保護隱私。")
            current_state = STATE_MAIN_MENU

    # --- 活體偵測 (眨眼判定) ---
    if current_state != STATE_MAIN_MENU and not has_blinked and len(face_locations) > 0:
        landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
        for landmarks in landmarks_list:
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                left_ear = calculate_ear(landmarks['left_eye'])
                right_ear = calculate_ear(landmarks['right_eye'])
                avg_ear = (left_ear + right_ear) / 2.0
                if avg_ear < EAR_THRESHOLD:  # 偵測到眨眼
                    has_blinked = True
                    print("\n👀 活體偵測通過！已確認為真人。")

    # --- UI 介面繪製 ---
    if current_state == STATE_MAIN_MENU:
        cv2.putText(frame, "MENU: 's' Store | 'o' Open | 'a' Admin | 'q' Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    else:
        mode_text = "[ STORE ]" if current_state == STATE_CONFIRM_STORE else "[ OPEN ]"
        cv2.putText(frame, f"{mode_text} 'y' Yes | 'n' No", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 顯示活體偵測狀態
        liveness_text = "Liveness: OK (Real Human)" if has_blinked else "Liveness: BLINK YOUR EYES!"
        liveness_color = (0, 255, 0) if has_blinked else (0, 0, 255)
        cv2.putText(frame, liveness_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, liveness_color, 2)

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left * 2, top * 2), (right * 2, bottom * 2), (255, 255, 0), 2)

    cv2.imshow('Smart Locker System', frame)
    key = cv2.waitKey(1) & 0xFF

    # --- 鍵盤監聽與操作 ---
    if current_state == STATE_MAIN_MENU:
        if key == ord('s'):
            current_state = STATE_CONFIRM_STORE
            state_start_time = time.time()
            has_blinked = False
            print("\n➡️ 進入存物模式：請看著鏡頭「眨眼」，然後按 [y] 確認。")
        elif key == ord('o'):
            if len(active_lockers) == 0:
                print("\n⚠️ 目前沒有任何存物紀錄，無法取物。")
            else:
                current_state = STATE_CONFIRM_OPEN
                state_start_time = time.time()
                has_blinked = False
                print("\n➡️ 進入取物模式：請看著鏡頭「眨眼」，然後按 [y] 解鎖。")

        # --- 隱藏管理員後臺 ---
        elif key == ord('a'):
            print("\n" + "=" * 40)
            pwd = input("🔐 [管理員後臺] 請在下方輸入超級密碼: ")
            if pwd == ADMIN_PWD:
                print(f"📦 目前佔用中的置物櫃: {list(active_lockers.keys())}")
                target = input("👉 請輸入要強制開啟的櫃號 (輸入 q 取消): ")
                if target.isdigit() and int(target) in active_lockers:
                    t_id = int(target)
                    trigger_door(t_id)
                    shutil.rmtree(os.path.join(base_dir, str(t_id)), ignore_errors=True)
                    del active_lockers[t_id]
                    print(f"🗑️ 已強制解鎖並清空第 {t_id} 號櫃。")
                else:
                    print("❌ 操作取消，或櫃號無效。")
            else:
                print("❌ 密碼錯誤！")
            print("=" * 40 + "\n")

        elif key == ord('q'):
            break

    elif current_state in [STATE_CONFIRM_STORE, STATE_CONFIRM_OPEN]:
        if key == ord('y'):
            if not has_blinked:
                print("\n⚠️ 活體偵測未通過：請對著鏡頭「眨眼」證明您是真人，再按 y！")
            else:
                if current_state == STATE_CONFIRM_STORE:
                    process_store(frame, face_encodings)
                else:
                    process_retrieve(face_encodings)
        elif key == ord('n'):
            current_state = STATE_MAIN_MENU
            print("\n↩️ 操作已取消，返回主選單。")

video_capture.release()
cv2.destroyAllWindows()
print("🛑 系統已安全關閉。")