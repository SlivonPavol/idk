import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import threading
from chrom import chrom
from fourier_analysis import estimate_heart_rate
from spo2 import spo2
import csv
import os
from datetime import datetime
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import requests

# Global variable to store the latest frame
latest_frame = None
heart_rate = []
oxygen_saturation = []

# Flask application
app = Flask(__name__)
socketio = SocketIO(app)  # Initialize SocketIO


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/data')
def data():
    # Read data from CSV file
    data = []
    with open('templates/data.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            data.append(row)

    # Pass data to the HTML template
    return render_template('data.html', data=data)

# CONSTANT
video_url = "http://espheartbeat.local:81/stream"
blur_radius = 5

# GLOBAL VARIABLES
first_10_seconds = False
heart_rate = []
oxygen_saturation = []
def send_frame(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def calculate_actual_fps(frame_count, prev_time):
    current_time = time.time()
    elapsed_time = current_time - prev_time
    actual_fps = 0

    if elapsed_time >= 1.0:
        actual_fps = frame_count / elapsed_time
        print(f"Actual FPS: {int(round(actual_fps)):}")
        frame_count = 0
        prev_time = current_time

    return frame_count, prev_time, actual_fps

def manage_csv_file(csv_file_path, new_data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Read existing data
    existing_data = []
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        with open(csv_file_path, 'r', newline='') as f:
            csv_reader = csv.reader(f)
            existing_data = list(csv_reader)

    # Insert new data at the beginning
    existing_data.insert(1, [timestamp] + new_data)

    # Write all data back to the file
    with open(csv_file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(existing_data)
def load_video_stream():
    video_capture  = cv2.VideoCapture(video_url)
    if not video_capture.isOpened():
        raise ValueError(f"Could not open video stream at {video_url}")
    return video_capture
def read_frame(video_capture):
    success, frame = video_capture.read()
    if not success:
        return None
    return frame
def gaussian_blur(image):
    return cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

def calculate_mean_rgb(blurred_image):
    indices_non_black = np.where(np.any(blurred_image != [0, 0, 0], axis=-1))
    non_black_pixels = blurred_image[indices_non_black]

    return np.mean(non_black_pixels[:, 0]), np.mean(non_black_pixels[:, 1]), np.mean(non_black_pixels[:, 2])

def calculate_hr_and_spo2(data, prw_fps):
    global first_10_seconds, heart_rate, oxygen_saturation

    channel_labels = ["B-channel", "G-channel", "R-channel"]
    np_results = np.array(data[:5])
    data_pd = pd.DataFrame(np_results)
    data_pd.columns = channel_labels
    signal_chrom = chrom(data, prw_fps, 20)

    heart_rate_bpm = estimate_heart_rate(signal_chrom, prw_fps)
    rounded_heart_rate = round(heart_rate_bpm * 60, 2)
    print(f"Heart Rate: {rounded_heart_rate} BPM")

    oxygen_percentage = spo2(data)
    rounded_oxygen_percentage = round(oxygen_percentage, 2)
    print(f"Oxygen Saturation (SpO2): {rounded_oxygen_percentage}%")
    if (first_10_seconds == False):
        # Emit heart rate and oxygen saturation data to clients via WebSocket
        socketio.emit('heart_rate', {'value': rounded_heart_rate})
        socketio.emit('oxygen_saturation', {'value': rounded_oxygen_percentage})
    first_10_seconds = True
    heart_rate.append(rounded_heart_rate)
    oxygen_saturation.append((rounded_oxygen_percentage))

def get_data(data, fps):
    if first_10_seconds:
        indices_of_newlines = [i for i, item in enumerate(data) if item == '\n'][:13]

        if indices_of_newlines:
            extracted_data = data[:indices_of_newlines[-1] + 1]

            extracted_data = [item for item in extracted_data if item != '\n']

            bgr_data = np.array(extracted_data)
        else:
            data = [item for item in data if item != '\n']
            bgr_data = np.array(data)

        prw_fps_rounded = np.mean(fps[:13])

    else:
        idk = [item for item in data if item != '\n']
        bgr_data = np.array(idk)
        prw_fps_rounded = np.mean(fps[:10])

    return bgr_data, prw_fps_rounded
def main():
    global latest_frame

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    rgb_data = []
    prw_fps = []
    frame_count = 0
    video_capture = load_video_stream()
    with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:
        start_time = time.time()
        data_timer = time.time()
        prev_time = time.time()
        while True:
            frame = read_frame(video_capture)
            if frame is None:
                continue
            height, width, _ = frame.shape
            frame = cv2.flip(frame, 1)
            latest_frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame.flags.writeable = False

            processed_img = face_mesh.process(frame)

            if processed_img.multi_face_landmarks:
                for face_landmarks in processed_img.multi_face_landmarks:
                    landmark_points = []
                    for i in range(0, 468):
                        x = int(face_landmarks.landmark[i].x * width)
                        y = int(face_landmarks.landmark[i].y * height)
                        landmark_points.append([x, y])

                    face = np.array((
                        landmark_points[10], landmark_points[338], landmark_points[297], landmark_points[332],
                        landmark_points[284], landmark_points[251],
                        landmark_points[389], landmark_points[435], landmark_points[367],
                        landmark_points[365], landmark_points[379], landmark_points[378],
                        landmark_points[400], landmark_points[377], landmark_points[152],
                        landmark_points[148], landmark_points[176], landmark_points[149], landmark_points[150],
                        landmark_points[136], landmark_points[138], landmark_points[177],
                        landmark_points[137], landmark_points[162], landmark_points[54], landmark_points[103],
                        landmark_points[67], landmark_points[109]))

                    eyes = np.array((
                        landmark_points[225], landmark_points[224], landmark_points[223], landmark_points[222],
                        landmark_points[221], landmark_points[193],
                        landmark_points[168], landmark_points[417], landmark_points[441],
                        landmark_points[442], landmark_points[443], landmark_points[444],
                        landmark_points[445], landmark_points[342], landmark_points[446],
                        landmark_points[261], landmark_points[448], landmark_points[449], landmark_points[450],
                        landmark_points[451], landmark_points[452], landmark_points[453],
                        landmark_points[464], landmark_points[465], landmark_points[351], landmark_points[6],
                        landmark_points[122], landmark_points[245], landmark_points[244], landmark_points[233],
                        landmark_points[232], landmark_points[231], landmark_points[230], landmark_points[229],
                        landmark_points[228], landmark_points[31], landmark_points[226], landmark_points[113]))

                    face_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(face_mask, [face], (255))

                    eyes_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(eyes_mask, [eyes], (255))

                    final_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(eyes_mask))

                    cropped_img = cv2.bitwise_and(frame, frame, mask=final_mask)

                    blurred_image = gaussian_blur(cropped_img)
                    mean_blue_blur, mean_green_blur, mean_red_blur = calculate_mean_rgb(blurred_image)

                    # Make a modifiable copy of the image array
                    frame_copy = np.copy(frame)

                    # Draw landmarks on the modifiable copy of the image array
                    mp_drawing.draw_landmarks(
                        image=frame_copy,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

                    # cv2.polylines(frame, [face], True, (0, 255, 255), 2)

                    #cv2.imshow('MediaPipe FaceMesh', frame)
                    #cv2.imshow('MediaPipe Masked pixel crop', cropped_img)

                    frame_count, prev_time, actual_fps = calculate_actual_fps(frame_count + 1, prev_time)
                    if actual_fps > 1:
                        prw_fps.append(int(round(actual_fps)))

                    rgb_data.append([mean_blue_blur, mean_green_blur, mean_red_blur])

                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 10:
                        if not first_10_seconds:
                            new_data, rounded_fps = get_data(rgb_data, prw_fps)
                            hr_thread = threading.Thread(target=calculate_hr_and_spo2, args=(new_data, rounded_fps))
                            hr_thread.start()
                            data_timer = time.time()
                            start_time = time.time()
                            if(prw_fps):
                                prw_fps.pop(0)
                        else:
                            if(heart_rate):
                                socketio.emit('heart_rate', {'value': round(np.mean(heart_rate), 2)})
                                socketio.emit('oxygen_saturation', {'value': round(np.mean(oxygen_saturation), 2)})
                                manage_csv_file('templates/data.csv',
                                                [round(np.mean(heart_rate)), round(np.mean(oxygen_saturation))])
                                url = 'http://espheartbeat.local:81/upload'  # Update the URL accordingly
                                payload = {
                                           "hr_fourier_chrom": round(np.mean(heart_rate), 2)
                                }
                                try:
                                   response = requests.post(url, json=payload)
                                   if response.status_code == 200:
                                     print("Data successfully sent to the endpoint")
                                except Exception as e:
                                   print("k")
                            heart_rate.clear()
                            oxygen_saturation.clear()
                            start_time = time.time()

                    if time.time() - data_timer >= 1:
                        data_timer = time.time()
                        rgb_data.append('\n')
                        if first_10_seconds:
                            index_of_newline = rgb_data.index('\n') if '\n' in rgb_data else -1
                            if index_of_newline != -1:
                                del rgb_data[:index_of_newline + 1]

                            new_data, rounded_fps = get_data(rgb_data, prw_fps)
                            hr_thread = threading.Thread(target=calculate_hr_and_spo2, args=(new_data, rounded_fps))
                            hr_thread.start()
                            if prw_fps:
                                prw_fps.pop(0)
                            data_timer = time.time()
            else:
                continue
                #cv2.imshow('MediaPipe FaceMesh', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    video_capture.release()


if __name__ == '__main__':
    # Start Flask app in a separate thread
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()

    # Start main function
    main()
