import cv2
import numpy as np

# Initialize camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for different colors
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([70, 255, 255])
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 30])
    blue_lower = np.array([100, 100, 100])
    blue_upper = np.array([130, 255, 255])
    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([20, 255, 255])
    purple_lower = np.array([130, 100, 100])
    purple_upper = np.array([160, 255, 255])

    # Threshold the HSV frames to get only specified colors
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
    black_mask = cv2.inRange(hsv_frame, black_lower, black_upper)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)
    purple_mask = cv2.inRange(hsv_frame, purple_lower, purple_upper)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours for different colors
    colors = {
        'Red': (red_mask, (0, 0, 255)),
        'Yellow': (yellow_mask, (0, 255, 255)),
        'Green': (green_mask, (0, 255, 0)),
        'Black': (black_mask, (0, 0, 0)),
        'Blue': (blue_mask, (255, 0, 0)),
        'Orange': (orange_mask, (0, 165, 255)),
        'Purple': (purple_mask, (255, 0, 255))
    }

    for color_name, (mask, color_value) in colors.items():
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_value, 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_value, 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()
