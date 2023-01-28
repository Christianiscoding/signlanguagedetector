import cv2

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Display the frame on the screen
    cv2.imshow("Webcam", frame)


    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
