from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import threading
from chrom import chrom
from fourier_analysis import estimate_heart_rate
from spo2 import spo2
import cProfile
from flask_socketio import SocketIO, emit
import csv
import os
from datetime import datetime

global disconnected

app = Flask(__name__)
socketio = SocketIO(app)  # Initialize SocketIO
video_url = "http://heartbeat.local:81/stream"
blur_radius = 5
rgb_data = []
temp = 0
extracted_data = None
heart_rate = []
oxygen_saturation = []

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

    # Open the CSV file in append mode with a specified newline character
    with open(csv_file_path, 'a', newline='') as f:
        csv_writer = csv.writer(f)

        # If the file is empty, write the headers
        if os.stat(csv_file_path).st_size == 0:
            csv_writer.writerow(['Timestamp', 'Heart Rate', 'Oxygen Saturation'])

        # Append the new value with the timestamp
        csv_writer.writerow([timestamp] + new_data)
def load_video_stream():
    video_capture  = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise ValueError(f"Could not open video stream at {video_url}")

    return video_capture

def gaussian_blur(image):
    return cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

def calculate_mean_rgb(blurred_image):
    indices_non_black = np.where(np.any(blurred_image != [0, 0, 0], axis=-1))
    non_black_pixels = blurred_image[indices_non_black]

    return np.mean(non_black_pixels[:, 0]), np.mean(non_black_pixels[:, 1]), np.mean(non_black_pixels[:, 2])

def calculate_hr_and_spo2(data, prw_fps):

    global temp, heart_rate, oxygen_saturation

    with cProfile.Profile() as pr:
        channel_labels = ["B-channel", "G-channel", "R-channel"]
        print(data)
        np_results = np.array(data[:5])
        data_pd = pd.DataFrame(np_results)
        data_pd.columns = channel_labels
        print(prw_fps)
        signal_chrom = chrom(data, prw_fps, 20)

        # Estimate heart rate in beats per minute (BPM)
        heart_rate_bpm = estimate_heart_rate(signal_chrom, prw_fps)
        rounded_heart_rate = round(heart_rate_bpm * 60, 2)
        print(f"Heart Rate: {rounded_heart_rate} BPM")

        # Calculate oxygen saturation (SpO2) percentage
        oxygen_percentage = spo2(data)
        rounded_oxygen_percentage = round(oxygen_percentage, 2)
        print(f"Oxygen Saturation (SpO2): {rounded_oxygen_percentage}%")
        if(temp == 0):
            # Emit heart rate and oxygen saturation data to clients via WebSocket
            socketio.emit('heart_rate', {'value': rounded_heart_rate})
            socketio.emit('oxygen_saturation', {'value': rounded_oxygen_percentage})
        temp = 1
        heart_rate.append(rounded_heart_rate)
        oxygen_saturation.append((rounded_oxygen_percentage))

    #pr.print_stats()
def get_data(data, fps):
    if temp == 1:
        # Find the indices of the first 13 occurrences of '\n'
        indices_of_newlines = [i for i, item in enumerate(data) if item == '\n'][:13]

        if indices_of_newlines:
            # Copy everything up to the 13th '\n'
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
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # Data buffers and timers (reset on new connection or timeout)
    rgb_data = []
    prw_fps = []
    frame_count = 0
    prev_time = time.time()
    start_time = time.time()
    last_client_connect = time.time()  # Track last client connection time
    connection_timeout = 5  # Adjust timeout as needed
    video_capture = load_video_stream()
    average_channel_values = np.zeros(10)
    with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:
        start_time = time.time()
        update_time = time.time()
        data_timer = time.time()
        last_client_connect = time.time()  # Track last client connection time
        connection_timeout = 5  # Adjust timeout as needed
        while True:
            success, frame = video_capture.read()
            if not success:
                continue
            height, width, _ = frame.shape
            # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert the RGB image back to BGR.

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

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

                    cv2.polylines(frame, [face], True, (0, 255, 255), 2)

                    #cv2.imshow('MediaPipe FaceMesh', frame)
                    #cv2.imshow('MediaPipe Masked pixel crop', cropped_img)

                    frame_count, prev_time, actual_fps = calculate_actual_fps(frame_count + 1, prev_time)
                    if actual_fps > 1:
                        prw_fps.append(int(round(actual_fps)))

                    rgb_data.append([mean_blue_blur, mean_green_blur, mean_red_blur])

                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 10:
                        if temp == 0:
                            new_data, rounded_fps = get_data(rgb_data, prw_fps)
                            hr_thread = threading.Thread(target=calculate_hr_and_spo2, args=(new_data, rounded_fps))
                            hr_thread.start()
                            data_timer = time.time()  # Reset the start time
                            start_time = time.time()  # Reset the start time
                            if(prw_fps):
                                prw_fps.pop(0)
                        else:
                            # Emit heart rate and oxygen saturation data to clients via WebSocket
                            if(heart_rate):
                                socketio.emit('heart_rate', {'value': round(np.mean(heart_rate), 2)})
                                socketio.emit('oxygen_saturation', {'value': round(np.mean(oxygen_saturation), 2)})
                                manage_csv_file('templates/data.csv',
                                                [round(np.mean(heart_rate)), round(np.mean(oxygen_saturation))])
                            heart_rate.clear()
                            oxygen_saturation.clear()
                            start_time = time.time()  # Reset the start time
                    if time.time() - update_time >= 1:
                        update_time = time.time()  # Reset the start time
                        rgb_data.append('\n')
                    if time.time() - data_timer >= 1:
                        if temp == 1:
                            # Find the index of the first occurrence of '\n'
                            index_of_newline = rgb_data.index('\n') if '\n' in rgb_data else -1

                            # Remove everything up to the first '\n'
                            if index_of_newline != -1:
                                rgb_data = rgb_data[index_of_newline + 1:]

                            new_data, rounded_fps = get_data(rgb_data, prw_fps)
                            hr_thread = threading.Thread(target=calculate_hr_and_spo2, args=(new_data, rounded_fps))
                            hr_thread.start()
                            if(prw_fps):
                                prw_fps.pop(0)
                            data_timer = time.time()  # Reset the start time

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



            else:
                cv2.imshow('MediaPipe FaceMesh', frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
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

# WebSocket route to handle client connections
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send initial heart rate and oxygen saturation values to the connected client
    socketio.emit('heart_rate', {'value': 0})
    socketio.emit('oxygen_saturation', {'value': 0})
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)



