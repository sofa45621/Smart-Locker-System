import cv2
import face_recognition
import os
import serial
import time
import shutil

# ==========================================
# 1. 初始化與環境設定
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

video_url = "http://192.168.0.96:4747/video"
video_capture = cv2.VideoCapture(video_url)

# ==========================================
# 2. 系統狀態與動態資料庫 (支援多人同時存物)
# ==========================================
STATE_MAIN_MENU = 0  # 主選單：等待選擇 S 或 O
STATE_CONFIRM_STORE = 1  # 存物確認：按 Y 拍照存檔，N 取消
STATE_CONFIRM_OPEN = 2  # 取物確認：按 Y 辨識開鎖，N 取消

current_state = STATE_MAIN_MENU

# 動態資料庫：紀錄目前所有佔用中的櫃子與人臉特徵 { 櫃號: 特徵編碼 }
active_lockers = {}

# 啟動時自動讀取硬碟，恢復之前的存物紀錄
print("⏳ 正在同步置物櫃資料...")
for folder_name in os.listdir(base_dir):
    if folder_name.isdigit():
        photo_path = os.path.join(base_dir, folder_name, "face.jpg")
        if os.path.exists(photo_path):
            img = face_recognition.load_image_file(photo_path)
            encs = face_recognition.face_encodings(img)
            if encs:
                active_lockers[int(folder_name)] = encs[0]
print(f"📦 目前使用中的置物櫃數量: {len(active_lockers)} 個")


def get_next_id():
    return max(active_lockers.keys()) + 1 if active_lockers else 1


def trigger_door():
    if ser:
        try:
            ser.write(b'1')
            print("⚡ 已發送開鎖指令給 STM32 (3秒後自動上鎖)。")
        except Exception as e:
            print(f"⚠️ 序列埠傳送失敗: {e}")
    else:
        print("⚡ (模擬) 門已開啟。")


# ==========================================
# 3. 核心邏輯
# ==========================================
def process_store(frame, face_encodings):
    global current_state
    if len(face_encodings) == 1:
        new_id = get_next_id()
        user_folder = os.path.join(base_dir, str(new_id))
        os.makedirs(user_folder, exist_ok=True)

        cv2.imwrite(os.path.join(user_folder, "face.jpg"), frame)
        active_lockers[new_id] = face_encodings[0]  # 寫入記憶體

        print(f"\n✅ 註冊成功！您是第 {new_id} 號客人。")
        trigger_door()
    elif len(face_encodings) == 0:
        print("\n⚠️ 畫面中找不到人臉，註冊失敗。")
    else:
        print("\n⚠️ 偵測到多張人臉，為確保安全，請單獨入鏡。")

    current_state = STATE_MAIN_MENU  # 做完動作回到主選單


def process_retrieve(face_encodings):
    global current_state
    if len(face_encodings) > 0:
        match_found = False
        matched_id = None

        # 跟所有正在使用中的櫃主比對
        for face_encoding in face_encodings:
            for locker_id, owner_encoding in active_lockers.items():
                match = face_recognition.compare_faces([owner_encoding], face_encoding, tolerance=0.45)
                if match[0]:
                    match_found = True
                    matched_id = locker_id
                    break
            if match_found: break

        if match_found:
            print(f"\n🔓 身分確認無誤！歡迎回來，第 {matched_id} 號客人。")
            trigger_door()

            # 刪除資料夾與記憶體紀錄 (閱後即焚)
            shutil.rmtree(os.path.join(base_dir, str(matched_id)), ignore_errors=True)
            del active_lockers[matched_id]
            print(f"🗑️ 第 {matched_id} 號客人的隱私資料已徹底銷毀。")
        else:
            print("\n❌ 取物失敗：系統中找不到符合您的存物紀錄！")
    else:
        print("\n⚠️ 畫面中找不到人臉，取物失敗。")

    current_state = STATE_MAIN_MENU  # 做完動作回到主選單


# ==========================================
# 4. 影像主迴圈與介面
# ==========================================
print("\n🎥 智慧置物櫃 Kiosk 系統啟動！")

while True:
    ret, frame = video_capture.read()
    if not ret: break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # --- 根據狀態顯示不同的 UI 文字 ---
    if current_state == STATE_MAIN_MENU:
        cv2.putText(frame, "[ MENU ] 's': Store | 'o': Open | 'q': Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
    elif current_state == STATE_CONFIRM_STORE:
        cv2.putText(frame, "[ STORE ] Look at camera -> 'y': Yes | 'n': No", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
    elif current_state == STATE_CONFIRM_OPEN:
        cv2.putText(frame, "[ OPEN ] Look at camera -> 'y': Yes | 'n': No", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    # 畫出人臉框
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left * 2, top * 2), (right * 2, bottom * 2), (255, 255, 0), 2)

    cv2.imshow('Smart Locker System', frame)
    key = cv2.waitKey(1) & 0xFF

    # --- 鍵盤監聽邏輯 ---
    if current_state == STATE_MAIN_MENU:
        if key == ord('s'):
            current_state = STATE_CONFIRM_STORE
            print("\n➡️ 進入存物模式：請正對鏡頭，按 [y] 確認拍照，按 [n] 取消。")
        elif key == ord('o'):
            if len(active_lockers) == 0:
                print("\n⚠️ 目前沒有任何存物紀錄，無法取物。")
            else:
                current_state = STATE_CONFIRM_OPEN
                print("\n➡️ 進入取物模式：請正對鏡頭，按 [y] 進行身分驗證，按 [n] 取消。")
        elif key == ord('q'):
            break

    elif current_state == STATE_CONFIRM_STORE:
        if key == ord('y'):
            process_store(frame, face_encodings)
        elif key == ord('n'):
            current_state = STATE_MAIN_MENU
            print("\n↩️ 已取消存物，返回主選單。")

    elif current_state == STATE_CONFIRM_OPEN:
        if key == ord('y'):
            process_retrieve(face_encodings)
        elif key == ord('n'):
            current_state = STATE_MAIN_MENU
            print("\n↩️ 已取消取物，返回主選單。")

video_capture.release()
cv2.destroyAllWindows()
print("🛑 系統已安全關閉。")