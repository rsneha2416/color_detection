import cv2
import numpy as np

# Define HSV color ranges for common colors
colors_hsv = {
    "Red": ([0, 100, 100], [10, 255, 255]),
    "Orange": ([11, 100, 100], [25, 255, 255]),
    "Yellow": ([26, 100, 100], [35, 255, 255]),
    "Green": ([36, 100, 100], [85, 255, 255]),
    "Cyan": ([86, 100, 100], [100, 255, 255]),
    "Blue": ([101, 100, 100], [130, 255, 255]),
    "Purple": ([131, 100, 100], [160, 255, 255]),
    "Pink": ([161, 100, 100], [170, 255, 255]),
    "White": ([0, 0, 200], [180, 30, 255]),
    "Gray": ([0, 0, 60], [180, 30, 200]),
    "Black": ([0, 0, 0], [180, 255, 60]),
}

# Function to map HSV value to a color name
def get_color_name(hsv_pixel):
    hsv_pixel_array = np.uint8([[hsv_pixel]])  # Shape: (1, 1, 3)
    for color_name, (lower, upper) in colors_hsv.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_pixel_array, lower_np, upper_np)
        if mask[0][0] == 255:
            return color_name
    return "Unknown"

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and get center pixel
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    pixel_bgr = frame[center_y, center_x]
    
    # Convert center pixel to HSV
    pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Get the color name from HSV
    color_name = get_color_name(pixel_hsv)

    # Draw center circle and color name
    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
    cv2.putText(frame, f"Color: {color_name}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show the frame
    cv2.imshow("Color Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
