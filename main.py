
import cv2
import mediapipe as mp
import numpy as np
import time
import websocket
import threading

position_horizontal = 0
position_vertical = 0

def calibration(mp_drawing, mp_pose, image, results, start_time):
    up = 440
    left = 640
    down = 300
    right = 500
    cv2.line(image, (int(1920 / 2), 0), (int(1920 / 2), 1080), (0, 0, 0), thickness=2)
    cv2.rectangle(image, (left, up), (right, down), color=(0, 0, 0), thickness=2)
    try:
        landmarks = results.pose_landmarks.landmark
    
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        x_middle = right_shoulder[0] + np.abs(left_shoulder[0] - right_shoulder[0]) / 2
        y_middle = left_shoulder[1] + np.abs(left_hip[1] - left_shoulder[1]) / 2
        
        putCircleAndText(image, x_middle, y_middle, (255, 255, 255))
        x_middle, y_middle = tuple(np.multiply([x_middle, y_middle], [1920, 1080]).astype(int))
        cv2.line(image, (0, y_middle - 75), (1920, y_middle - 75), (0, 0, 0), thickness=2)
        cv2.line(image, (0, y_middle + 75), (1920, y_middle + 75), (0, 0, 0), thickness=2)


        x_pos, y_pos = tuple(np.multiply([right_wrist[0], right_wrist[1]], [1920, 1080]).astype(int))

        if x_pos < left and x_pos > right and y_pos > down and y_pos < up:            
            if start_time is None:
                start_time = time.time()

            if time.time() - start_time > 3:
                y_line_jump = y_middle - 75
                y_line_squat = y_middle + 75
                return True, start_time, y_line_jump, y_line_squat
        else:
            start_time = None
            putCircleAndText(right_wrist[0], right_wrist[1], (0, 0, 0))  # TODO: To delete

    except:
        pass
         
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    return False, start_time, None, None


def play(mp_pose, mp_drawing, image, bufferSize, results, y_line_jump, y_line_squat):
    global position_horizontal
    global position_vertical
    cv2.line(image, (0, y_line_jump), (1920, y_line_jump), (0, 0, 0), thickness=2)
    cv2.line(image, (0, y_line_squat), (1920, y_line_squat), (0, 0, 0), thickness=2)

    cv2.rectangle(image, (0, 0), (640, 1080), color=(255, 255, 255), thickness=2)
    
    cv2.rectangle(image, (1280, 0), (1920, 1080), color=(255, 255, 255), thickness=2)
    try:
        landmarks = results.pose_landmarks.landmark
        
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        
        x_middle = right_shoulder[0] + np.abs(left_shoulder[0] - right_shoulder[0]) / 2
        y_middle = left_shoulder[1] + np.abs(left_hip[1] - left_shoulder[1]) / 2

        if position_horizontal == 1 and x_middle * 1920 > 640:
            ws.send("LEFT")
            position_horizontal = 0
        if position_horizontal == 0 and x_middle * 1920 > 1280:
            ws.send("LEFT")
            position_horizontal = -1

        if position_horizontal == -1 and x_middle * 1920 < 1280:
            ws.send("RIGHT")
            position_horizontal = 0
        if position_horizontal == 0 and x_middle * 1920 < 640:
            ws.send("RIGHT")
            position_horizontal = 1


        if y_middle * 1080 < y_line_jump and position_vertical == 0:
            position_vertical = 1
            ws.send("UP")
        if y_middle * 1080 > y_line_squat and position_vertical == 0:
            position_vertical = 1
            ws.send("DOWN")
        if y_middle * 1080 < y_line_squat and y_middle * 1080 > y_line_jump:
            position_vertical = 0
            
    except:
        pass
        
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
               

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


def putCircleAndText(image, x_middle, y_middle, color):
    cv2.putText(image, str(np.multiply([x_middle, y_middle], [1920, 1080])), 
                tuple(np.multiply([x_middle, y_middle], [1920, 1080]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    cv2.circle(image, tuple(np.multiply([x_middle, y_middle], [1920, 1080]).astype(int)), radius=5, color=color, thickness=10)


def main(ws):
    global position_horizontal
    global position_vertical
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    isSuccessful = False
    start_time = None
    y_line_jump = None
    y_line_squat = None

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            print(position_vertical)
            ret, frame = cap.read()
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            bufferSize = 50

            if (isSuccessful):
                play(mp_pose, mp_drawing, image, bufferSize, results, y_line_jump, y_line_squat)
            else:
                isSuccessful, start_time, y_line_jump, y_line_squat = calibration(mp_drawing, mp_pose, image, results, start_time)
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def get_user_input(ws):
    while True:
        main()
        user_input = input("Enter direction (UP/DOWN/LEFT/RIGHT): ")
        if user_input.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
            ws.send(user_input.upper())
        else:
            print("Invalid input. Please enter UP, DOWN, LEFT, or RIGHT.")

def on_open(ws):
    threading.Thread(target=main, args=(ws,)).start()


if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://localhost:8080/Auth")
    ws.on_open = on_open
    ws.run_forever()