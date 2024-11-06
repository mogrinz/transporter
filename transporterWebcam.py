import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
from PIL import Image
import pyvirtualcam
import os

#transporterWebcam, Nov 2024 Michael Ogrinz
#https://github.com/mogrinz/transporter

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Load the static background image
background_image = cv2.imread('background.png')
if background_image is None:
    print("Error: 'background.png' not found.")
    exit()

# Load animated GIF frames, excluding the first and last frames
def load_glitter_gif(glitter_path):
    try:
        glitter_gif = Image.open(glitter_path)
        gif_frames = []
        for frame in range(1, glitter_gif.n_frames - 1):  # Skip the first and last frame
            glitter_gif.seek(frame)
            gif_frame = glitter_gif.convert("RGB")
            gif_frame = np.array(glitter_gif)
            gif_frame = cv2.cvtColor(gif_frame, cv2.COLOR_RGB2BGR)
            gif_frame = cv2.resize(gif_frame, (background_image.shape[1], background_image.shape[0]))  # Ensure same size
            gif_frames.append(gif_frame)
        print(f"Loaded {len(gif_frames)} frames from '{glitter_path}', excluding the first and last frames.")
        return gif_frames
    except Exception as e:
        print(f"Error loading frames from '{glitter_path}': {e}")
        exit()

# Initialize the glitter GIF
effects_dir = "effects"
effect_files = [os.path.join(effects_dir, f) for f in os.listdir(effects_dir) if f.lower().endswith('.gif')]
current_effect_index = 0
gif_frames = load_glitter_gif(effect_files[current_effect_index])

# Initialize camera
camera_index = 0
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Error: Camera with index {camera_index} could not be opened.")
    exit()

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True)

# Duration of the effect in seconds
duration_seconds = 3
frame_rate = 10
alpha_increment = 0.1

# Function to display a 5-second countdown before capturing the background
def countdown_and_capture():
    countdown_seconds = 5
    while countdown_seconds > 0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (background_image.shape[1], background_image.shape[0]))

        text = str(countdown_seconds)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        color = (0, 0, 255)
        thickness = 10
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        cv2.imshow('Webcam Feed', frame)
        cv2.waitKey(1000)
        countdown_seconds -= 1

    for _ in range(15):
        cap.read()
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (background_image.shape[1], background_image.shape[0]))
        cv2.imwrite('background.png', frame)
        print("New background image captured and saved as background.png.")
        return frame

# Function to show a 5-second live preview from the currently active camera
def show_preview(seconds=5):
    start_time = time.time()
    while (time.time() - start_time) < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (background_image.shape[1], background_image.shape[0]))
        cv2.imshow('Live Preview', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Live Preview')

# Main loop flags
live_mode = False
show_background = True
transport_out_countdown_flag = False

# Function to switch to a specified camera with retries
def switch_camera(index, retries=3, delay=0.5):
    global cap, camera_index
    cap.release()
    camera_index = index
    for attempt in range(retries):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Switched to camera {camera_index}")
            return
        else:
            print(f"Attempt {attempt + 1} to access camera {camera_index} failed.")
            cap.release()
            time.sleep(delay)
    print(f"Failed to initialize camera {camera_index}. Reverting to camera 0.")
    camera_index = 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

def transport_out():
    print("Starting reverse transporter effect.")
                
    pygame.mixer.music.load("sound_effect.mp3")
    pygame.mixer.music.play()

    frame_index = 0
    glitter_alpha = 0.0

    while glitter_alpha < 0.9:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (background_image.shape[1], background_image.shape[0]))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.segmentation_mask is not None:
            silhouette_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            silhouette_mask = cv2.resize(silhouette_mask, (background_image.shape[1], background_image.shape[0]))

            gif_frame = gif_frames[frame_index % len(gif_frames)]
            masked_glitter = cv2.bitwise_and(gif_frame, gif_frame, mask=silhouette_mask)

            live_silhouette = cv2.bitwise_and(frame, frame, mask=silhouette_mask)
            combined_silhouette = cv2.addWeighted(masked_glitter, glitter_alpha, live_silhouette, 1 - glitter_alpha, 0)

            combined_frame = background_image.copy()
            combined_frame[silhouette_mask > 0] = combined_silhouette[silhouette_mask > 0]
                        
            cv2.imshow('Webcam Feed', combined_frame)
            cam.send(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(int(1000 / frame_rate))

            glitter_alpha += alpha_increment
            frame_index += 1

    fade_out_duration_ms = int((glitter_alpha / alpha_increment) * (1000 / frame_rate))
    pygame.mixer.music.fadeout(fade_out_duration_ms)

    while glitter_alpha > 0.0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (background_image.shape[1], background_image.shape[0]))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.segmentation_mask is not None:
            silhouette_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            silhouette_mask = cv2.resize(silhouette_mask, (background_image.shape[1], background_image.shape[0]))

            gif_frame = gif_frames[frame_index % len(gif_frames)]
            masked_glitter = cv2.bitwise_and(gif_frame, gif_frame, mask=silhouette_mask)

            combined_silhouette = cv2.addWeighted(masked_glitter, glitter_alpha, background_image, 1 - glitter_alpha, 0)
            combined_frame = background_image.copy()
            combined_frame[silhouette_mask > 0] = combined_silhouette[silhouette_mask > 0]

            cv2.imshow('Webcam Feed', combined_frame)
            cam.send(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(int(1000 / frame_rate))

            glitter_alpha -= alpha_increment
            frame_index += 1

    cv2.imshow('Webcam Feed', background_image)
    cam.send(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1000)


# Start virtual camera with PyVirtualCam
with pyvirtualcam.Camera(width=background_image.shape[1], height=background_image.shape[0], fps=30) as cam:
    print(f"Using virtual camera: {cam.device}")
    while True:
        if show_background:
            #cam.send(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
            cv2.imshow('Webcam Feed', background_image)
            key = cv2.waitKey(100)

            if key == ord('b'):
                background_image = countdown_and_capture()
            elif key == ord('l') and not live_mode:
                # Immediately go into live mode if 'L' key is pressed
                print("Switching to live mode immediately.")
                live_mode = True
                show_background = False
            elif key == ord('q'):
                print("Exiting program.")
                break
            elif key == ord('p'):
                print("Showing 5-second live preview.")
                show_preview(5)
            elif key == ord('n'):
                current_effect_index = (current_effect_index + 1) % len(effect_files)
                gif_frames = load_glitter_gif(effect_files[current_effect_index])
                print(f"Loaded effect: {effect_files[current_effect_index]}")
            elif key in [ord(str(i)) for i in range(5)]:
                switch_camera(int(chr(key)))
            elif key == ord('t'):
                print("Starting transporter effect.")
                time.sleep(3)
                pygame.mixer.music.load("sound_effect.mp3")
                pygame.mixer.music.play()

                frame_index = 0
                glitter_alpha = 0.0
                silhouette_alpha = 0.0

                while glitter_alpha < 0.9:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (background_image.shape[1], background_image.shape[0]))

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.segmentation_mask is not None:
                        silhouette_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
                        silhouette_mask = cv2.resize(silhouette_mask, (background_image.shape[1], background_image.shape[0]))

                        gif_frame = gif_frames[frame_index % len(gif_frames)]
                        masked_glitter = cv2.bitwise_and(gif_frame, gif_frame, mask=silhouette_mask)

                        combined_frame = background_image.copy()
                        combined_frame[silhouette_mask > 0] = cv2.addWeighted(masked_glitter, glitter_alpha, background_image, 1 - glitter_alpha, 0)[silhouette_mask > 0]
                        
                        cv2.imshow('Webcam Feed', combined_frame)
                        cam.send(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
                        cv2.waitKey(int(1000 / frame_rate))

                        glitter_alpha += alpha_increment
                        frame_index += 1

                while silhouette_alpha < 1.0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (background_image.shape[1], background_image.shape[0]))

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.segmentation_mask is not None:
                        silhouette_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
                        silhouette_mask = cv2.resize(silhouette_mask, (background_image.shape[1], background_image.shape[0]))

                        gif_frame = gif_frames[frame_index % len(gif_frames)]
                        masked_glitter = cv2.bitwise_and(gif_frame, gif_frame, mask=silhouette_mask)
                        live_silhouette = cv2.bitwise_and(frame, frame, mask=silhouette_mask)

                        combined_silhouette = cv2.addWeighted(masked_glitter, 0.9 - silhouette_alpha, live_silhouette, silhouette_alpha, 0)
                        combined_frame = background_image.copy()
                        combined_frame[silhouette_mask > 0] = combined_silhouette[silhouette_mask > 0]
                        
                        cv2.imshow('Webcam Feed', combined_frame)
                        cam.send(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
                        cv2.waitKey(int(1000 / frame_rate))

                        silhouette_alpha += alpha_increment
                        frame_index += 1

                live_mode = True
                show_background = False
                pygame.mixer.music.fadeout(1000)

        elif live_mode:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (background_image.shape[1], background_image.shape[0]))
            cv2.imshow('Webcam Feed', frame)
            cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(30)
            
            
            if transport_out_countdown_flag==True and (time.time() - start_time) > 3:
                transport_out()
                transport_out_countdown_flag=False
                show_background = True
                live_mode = False
            
            if key == ord('o'):
                print("Starting 3-second delay before transport-out effect.")
                transport_out_countdown_flag = True
                start_time = time.time()

            if key == ord('q'):
                print("Exiting program.")
                break
            elif key == ord('n'):
                current_effect_index = (current_effect_index + 1) % len(effect_files)
                gif_frames = load_glitter_gif(effect_files[current_effect_index])
                print(f"Loaded effect: {effect_files[current_effect_index]}")

cap.release()
cv2.destroyAllWindows()
pose.close()
pygame.mixer.quit()
