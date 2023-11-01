import cv2


def take_image():
    # Initialize the webcam
    global ret
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (usually the built-in webcam)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
    else:
        # Capture a single frame
        ret, frame_x = cap.read()
        for i in range(10):
            ret, frame = cap.read()
            frame = (frame_x + frame)/2
            frame_x = frame

        if not ret:
            print("Error: Could not capture a frame.")
        else:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # Save the captured frame as an image file
            cv2.imwrite("captured_image.jpg", frame)
            print("Image saved as 'captured_image.jpg'")

        # Release the webcam
        cap.release()

    # Close any OpenCV windows that may have opened during the process
    cv2.destroyAllWindows()


take_image()
