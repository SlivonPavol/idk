import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import threading
from chrom import chrom
from fourier_analysis import estimate_heart_rate
from spo2 import spo2


class Heartbeat:
    def __init__(self):
        self.video_url = "http://heartbeat.local:81/stream"
        self.blur_radius = 5
        self.rgb_data = []

    def calculate_actual_fps(self, frame_count, prev_time):
        current_time = time.time()
        elapsed_time = current_time - prev_time
        actual_fps = 0

        if elapsed_time >= 1.0:
            actual_fps = frame_count / elapsed_time
            print(f"Actual FPS: {actual_fps:.2f}")
            frame_count = 0
            prev_time = current_time

        return frame_count, prev_time, actual_fps

    def load_video_stream(self):
        video_capture  = cv2.VideoCapture(self.video_url)
        if not video_capture.isOpened():
            raise ValueError(f"Could not open video stream at {self.video_url}")

        return video_capture

    def gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (self.blur_radius, self.blur_radius), 0)

    def calculate_mean_rgb(self, blurred_image):
        indices_non_black = np.where(np.any(blurred_image != [0, 0, 0], axis=-1))
        non_black_pixels = blurred_image[indices_non_black]

        return np.mean(non_black_pixels[:, 0]), np.mean(non_black_pixels[:, 1]), np.mean(non_black_pixels[:, 2])


    def calculate_hr_and_spo2(self, rgb_data, prw_fps):
        bgr_data = np.array(rgb_data)
        self.rgb_data = []

        channel_labels = ["B-channel", "G-channel", "R-channel"]
        np_results = np.array(bgr_data[:5])
        data_pd = pd.DataFrame(np_results)
        data_pd.columns = channel_labels

        avg_fps = np.nanmean(prw_fps)
        prw_fps_rounded = round(avg_fps)

        signal_chrom = chrom(bgr_data, prw_fps_rounded, 20)

        # Estimate heart rate in beats per minute (BPM)
        heart_rate_bpm = estimate_heart_rate(signal_chrom, prw_fps_rounded)
        rounded_heart_rate = round(heart_rate_bpm * 60, 2)
        print(f"Heart Rate: {rounded_heart_rate} BPM")

        # Calculate oxygen saturation (SpO2) percentage
        oxygen_percentage = spo2(bgr_data)
        rounded_oxygen_percentage = round(oxygen_percentage, 2)
        print(f"Oxygen Saturation (SpO2): {rounded_oxygen_percentage}%")


    def main(self):
        prw_fps = []
        rgb_data = []
        frame_count = 0
        prev_time = time.time()
        video_capture = self.load_video_stream()
        average_channel_values = np.zeros(10)
        start_time = time.time()
        while True:
            # Read a frame from the camera
            success, frame = video_capture.read()
            if not success:
                continue

            height, width, _ = frame.shape
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mask = np.zeros_like(frame)

            # Define the region of interest (x, y, width, height)
            #roi = (200, 50, 100, 100)
            roi = (200, 50, 75, 200)

            # Draw a rectangle around the region of interest
            x, y, w, h = roi
            # Create a mask for the region of interest
            mask[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
            cropped_image = cv2.bitwise_and(frame, mask)
            blurred_image = self.gaussian_blur(cropped_image)
            mean_blue_blur, mean_green_blur, mean_red_blur = self.calculate_mean_rgb(blurred_image)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the original frame
            cv2.imshow('Original Frame', frame)

            # Display the frame with the masked region
            cv2.imshow('Video with Masked Region', blurred_image)

            frame_count, prev_time, actual_fps = self.calculate_actual_fps(frame_count + 1, prev_time)
            if actual_fps > 1:
                prw_fps.append(actual_fps)

            rgb_data.append([mean_blue_blur, mean_green_blur, mean_red_blur])

            elapsed_time = time.time() - start_time
            if elapsed_time >= 10:
                hr_thread = threading.Thread(target=self.calculate_hr_and_spo2, args=(rgb_data, prw_fps))
                hr_thread.start()
                rgb_data.clear()
                prw_fps.clear()

                start_time = time.time()  # Reset the start time

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close the window
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    heartbeat = Heartbeat()
    heartbeat.main()
