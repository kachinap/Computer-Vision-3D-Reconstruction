import cv2
import numpy as np

# CHOICE 2: Automating the way the thresholds are determined
# Function that builds a background model by averaging the first num_frames of the background video without using built-in background subtraction functions
def build_background_model_average(bg_video_path, num_frames=30):
    cap = cv2.VideoCapture(bg_video_path)
    sum_frame = None
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame.astype(np.float32)
        if sum_frame is None:
            sum_frame = frame
        else:
            sum_frame += frame
        count += 1
    cap.release()
    if count == 0:
        return None
    bg_frame = (sum_frame / count).astype(np.uint8)
    return bg_frame

# Function that computes the foreground mask by comparing the current frame to the background model
def get_foreground_mask(frame, bg_bgr, min_blob_area=500):
    # Apply Gaussian blur to reduce noise
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    bg_blur = cv2.GaussianBlur(bg_bgr, (5, 5), 0)
    
    # Convert images to HSV color space
    frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(bg_blur, cv2.COLOR_BGR2HSV)
    
    # Compute the absolute difference between current frame and background
    diff = cv2.absdiff(frame_hsv, bg_hsv)
    
    # Automatically determine threshold for each channel using Otsu's method
    _, mask_h = cv2.threshold(diff[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_s = cv2.threshold(diff[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_v = cv2.threshold(diff[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine the masks from all channels
    combined_mask = cv2.bitwise_or(mask_h, mask_s)
    combined_mask = cv2.bitwise_or(combined_mask, mask_v)
    
    # Refine the mask using morphological operations and median blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed_mask = cv2.medianBlur(processed_mask, 5)
    
    # Remove small blobs based on contour area
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(processed_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_blob_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    
    return filtered_mask