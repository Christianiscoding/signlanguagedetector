import cv2

# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Apply background subtraction to the frame
    fg_mask = bg_subtractor.apply(frame)

    # Apply morphological operations to the foreground mask to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Display the foreground mask on the screen
    cv2.imshow("Foreground Mask", fg_mask)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()