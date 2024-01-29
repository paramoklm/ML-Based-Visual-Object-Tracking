import cv2
import numpy as np
from KalmanFilter import KalmanFilter
from Detector import detect

def main():
    # Create Kalman filter object
    dt = 0.1
    u_x, u_y = 1, 1
    std_acc = 1
    x_dt_meas, y_dt_meas = 0.1, 0.1
    kalman_filter = KalmanFilter(dt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas)

    # Create video capture object
    cap = cv2.VideoCapture('./randomball.avi')  # Replace with the actual path

    start = True

    trajectory = []

    while cap.isOpened():
        if start:
            ret, frame = cap.read()
            start = False
            continue

        if not ret:
            break

        # Object Detection
        centers = detect(frame)

        if centers:
            # Track the object using Kalman Filter
            print("Centers 1: ", centers)
            kalman_filter.predict()
            print("Centers 2: ", centers)
            kalman_filter.update(np.array(centers[0]))
            print("Centers 3: ", centers)

            # Visualize tracking results
            detected_center = centers[0].astype(int).flatten()
            print("Detected Centers: ", detected_center)
            predicted_position = kalman_filter.x_pred[:2].astype(int).flatten()
            print("Predicted Position: ", predicted_position)

            if not cap.isOpened():
                break

            ret, frame_next = cap.read()

            if not ret:
                break

            centers_next = detect(frame_next)

            detected_center_next = centers_next[0].astype(int).flatten()

            middle_point = (predicted_position + detected_center_next) / 2
            middle_point = middle_point.astype(int)

            trajectory.append(tuple(middle_point))
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (147, 20, 255), 2)

            # Handle different OpenCV versions
            if cv2.__version__.startswith('3'):
                cv2.circle(frame, tuple(detected_center), 5, (0, 255, 0), -1)  # Detected circle
            elif cv2.__version__.startswith('4'):
                cv2.circle(frame, tuple(detected_center), 5, (0, 255, 0), -1)  # Detected circle

            cv2.rectangle(frame, tuple(middle_point - 5), tuple(middle_point + 5), (255, 0, 0), 2)  # Predicted rectangle
            cv2.rectangle(frame, tuple(predicted_position - 5), tuple(predicted_position + 5), (0, 0, 255), 2)  # Estimated rectangle
            cv2.imshow('Object Tracking', frame)
            if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
                break

            frame = frame_next

    cap.release()
    cv2.destroyAllWindows()

    # Create an empty image for the trajectory
    trajectory_img = np.zeros_like(frame)

    # Draw the trajectory on the image
    for i in range(1, len(trajectory)):
        cv2.line(trajectory_img, trajectory[i - 1], trajectory[i], (147, 20, 255), 2)

    # Save the trajectory image
    cv2.imwrite('trajectory.png', trajectory_img)

if __name__ == "__main__":
    main()

