import cv2

class VideoCapture:
    def __init__(self, camera_id=0):
        # This is an instance attribute for capturing video
        self.cap = cv2.VideoCapture(camera_id)

    def show_video_stream(self):
        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Display the resulting frame
            cv2.imshow('Video Stream', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture and destroy the windows
        self.cap.release()
        cv2.destroyAllWindows()


