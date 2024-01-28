import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter

# Function to draw bounding boxes and track IDs on frames
def draw_boxes_on_frames(detections, output_path):
    for frame_num, frame_detection in detections.items():
        # Load the frame
        frame_path = f"../ADL-Rundle-6/img1/{frame_num:06d}.jpg"  # Replace with the actual path to your frames
        frame = cv2.imread(frame_path)

        # Draw bounding boxes and track IDs on the frame
        for detection in frame_detection:
            track_id, x, y, w, h, _, _, _, _ = detection
            #if track_id != 1:
            #    continue
            x, y, w, h = map(int, [x, y, w, h])
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display track ID
            cv2.putText(frame, str(int(track_id)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the frame with bounding boxes and track IDs
        output_frame_path = f"{output_path}/{frame_num:06d}_with_boxes.jpg"
        cv2.imwrite(output_frame_path, frame)

# Function to load detections from a text file
def load_detections(file_path):
    detections_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            frame = int(values[0])
            detection = np.array(list(map(float, values[1:])))

            if frame not in detections_dict:
                detections_dict[frame] = []

            detections_dict[frame].append(detection)

    # Convert detections to NumPy arrays
    for frame, detections in detections_dict.items():
        detections_dict[frame] = np.array(detections)

    return detections_dict


# Function to compute IoU (Jaccard Index)
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = w1 * h1 + w2 * h2 - intersection_area
    iou = intersection_area / union_area

    return iou

def bbox_to_centroides(left, top, width, height):
    x_center = left + 0.5 * width
    y_center = top + 0.5 * height

    return x_center, y_center

def centroides_to_bbox(x_center, y_center, width, height):
    x = int(x_center - width / 2)
    y = int(y_center - height / 2)

    return [x, y, width, height]

def init_kalman_filter(left, top, width, height):
    dt = 0.1
    u_x, u_y = 0.1, 0.1
    std_acc = 1
    x_sdt_meas, y_sdt_meas = 1, 1
    kalman_filter = KalmanFilter(dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas)
    kalman_filter.x_pred = np.array([[left], [top], [width], [height]])

    return kalman_filter

def update_kalman_filter(kalman_filter, left, top, width, height):
    x, y = get_bbox_center(left, top, width, height)
    z_measurement = np.array([[x], [y]])
    kalman_filter.update(z_measurement)

    return kalman_filter.x_pred.flatten().tolist()

def predict_kalman_filter(kalman_filter, width, height):

    kalman_filter.predict()
    x, y, _, _ = kalman_filter.x_pred.flatten().tolist()
    return centroides_to_bbox(x, y, width, height)

def match_to_track(detections, tracks, karman_filters, frame, track_count, sigma_iou):
    # Init matrix with computations
    cost_matrix = np.zeros((len(tracks), len(detections[frame])))

    # Compute iou for all combinations
    # For all tracks
    for i, (track_id, track) in enumerate(tracks.items()):
        # Get last bbox of track
        obj_id_t, x_t, y_t, w_t, h_t, conf_t, _, _, _ = track[-1]
        # Get karman_filter for track
        kalman_filter = kalman_filters[track_id]
        checkpoints = (kalman_filters.x_pred, kalman_filters.P)

        # Predict with kalman filter and specific bbox
        predict_bbox = predict_kalman_filter(kalman_filter, width, height)

        # Reset kalman filter to checkpoint
        kalman_filters.x_pred, kalman_filters.P = checkpoints

        # For all detections
        for j, detection in enumerate(detections[frame]):
            obj_id, x, y, w, h, conf, _, _, _ = detection
            current_box = [x, y, w, h]

            # Predict with kalman filter and specific bbox
            predict_bbox = predict_kalman_filter(kalman_filter, width, height)

            # Compute IoU between the last track detection and the current detection
            iou = calculate_iou(last_track_box, current_box)

            # Cost is the complement of IoU, as linear_sum_assignment finds the minimum cost assignment
            cost_matrix[i, j] = 1 - iou


    # Use Hungarian algorithm to find optimal assignment
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    tracks_tr_delete = []
    unmatched_det = []

     # Update tracks based on the optimal assignment
     # For all matched tracks index
    for i, track_id in enumerate(track_indices):
        # Get corresponding detection id
        detection_id = detection_indices[i]
        iou = 1 - cost_matrix[track_id, detection_id]

        # If above treshold
        if iou >= sigma_iou:
            # Get track
            track = tracks[list(tracks.keys())[track_id]]

            # Get detection
            obj_id, x, y, w, h, conf, _, _, _ = detections[frame][detection_id]

            # Update filter
            update_kalman_filter(kalman_filters[list(tracks.keys())[track_id]], x, y, w, h)

            # Append detection to track
            track.append(detections[frame][detection_id])

            # Associate detection to track
            detections[frame][detection_id][0] = list(tracks.keys())[track_id]

        else:
            # Add detection to unmatched list
            unmatched_det.append(detection_id)

            # Add track to unmatched track
            tracks_tr_delete.append(list(tracks.keys())[track_id])

    # Delete tracks with no matches
    unmatched_tracks = set(range(len(tracks))) - set(track_indices)

    # For tracks to delete
    for delete_id in sorted(unmatched_tracks, reverse=True):
        # Delete
        tracks.pop(list(tracks.keys())[delete_id])

    # For tracks to delete because of threshold
    for delete_id in tracks_tr_delete:
        # Delete
        tracks.pop(delete_id)

    # Create new tracks for unmatched detections
    # For all detections in frame
    # TODO: why range ? possible fix
    for j, detection_id in enumerate(range(len(detections[frame]))):
        # If index not in matched detections
        if j not in detection_indices:
            # Associate detection to new track id
            detections[frame][detection_id][0] = track_count

            # Get detection
            obj_id, x, y, w, h, conf, _, _, _ = detections[frame][detection_id]

            # Init kalman filter
            new_kalman_filter = init_kalman_filter(x, y, w, h)

            # Add new track
            tracks[track_count] = [detections[frame][detection_id]]

            # Add corresponding kalman filter
            kalman_filters[track_count] = new_kalman_filter

            # Increment track_count
            track_count += 1


    # Create new tracks for unmatched detections because of threshold
    for detection in unmatched_det:

        # Associate detection to new track id
        detections[frame][detection][0] = track_count

        # Get detection
        obj_id, x, y, w, h, conf, _, _, _ = detections[frame][detection_id]

        # Init kalman filter
        new_kalman_filter = init_kalman_filter(x, y, w, h)

        # Add new track
        tracks[track_count] = [detections[frame][detection_id]]

        # Add corresponding kalman filter
        kalman_filters[track_count] = new_kalman_filter

        # Increment track_count
        track_count += 1

    return tracks, track_count

def main():
    # Load detections from the text file
    detections = load_detections("../ADL-Rundle-6/det/det.txt")

    # Initialize empty tracks dictionary and track count
    tracks = {}
    track_count = 0

    # Perform multi-object tracking using Hungarian algorithm
    for frame in range(1, len(detections) + 1):
        tracks, track_count = match_to_track(detections, tracks, frame, track_count, 0.01)

    # Draw bounding boxes and track IDs on frames
    output_frames_path = "output/"  # Replace with the desired output path
    draw_boxes_on_frames(detections, output_frames_path)

if __name__ == "__main__":
    main()
