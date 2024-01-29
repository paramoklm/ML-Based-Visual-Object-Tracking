import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter

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
def calculate_iou(box1, box2):
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

def get_bbox_center(left, top, width, height):
    x_center = left + 0.5 * width
    y_center = top + 0.5 * height

    return x_center, y_center

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

def predict_kalman_filter(kalman_filter):
    kalman_filter.predict()
    return kalman_filter.x_pred.flatten().tolist()


def match_to_track(detections, tracks, frame, track_count, sigma_iou):
    # Create a cost matrix to represent the dissimilarity between tracks and detections
    cost_matrix = np.zeros((len(tracks), len(detections[frame])))

    # Do all combinations
    # For all tracks
    for i, track in enumerate(tracks.values()):
        # Get 
        obj_id_t, x_t, y_t, w_t, h_t, conf_t, _, _, _ = track[-1][0]
        kalman_filter = track[0]
        last_track_box = predict_kalman_filter(kalman_filter) #[x_t, y_t, w_t, h_t]

        for j, detection in enumerate(detections[frame]):
            obj_id, x, y, w, h, conf, _, _, _ = detection
            current_box = [x, y, w, h]

            # Compute IoU between the last track detection and the current detection
            iou = calculate_iou(last_track_box, current_box)

            # Cost is the complement of IoU, as linear_sum_assignment finds the minimum cost assignment
            cost_matrix[i, j] = 1 - iou

    # Use Hungarian algorithm to find optimal assignment
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    tracks_tr_delete = []
    unmatched_det = []

    # Update tracks based on the optimal assignment
    for i, track_id in enumerate(track_indices):
        detection_id = detection_indices[i]
        iou = 1 - cost_matrix[track_id, detection_id]

        if iou >= sigma_iou:
            # If IoU is above threshold, assign the detection to the track
            _, track = tracks[list(tracks.keys())[track_id]]
            # TODO: Update kalman filter
            obj_id, x, y, w, h, conf, _, _, _ = detections[frame][detection_id]

            update_kalman_filter(tracks[list(tracks.keys())[track_id]][0], x, y, w, h)
            track.append(detections[frame][detection_id])
            detections[frame][detection_id][0] = list(tracks.keys())[track_id]
        else:
            unmatched_det.append(detection_id)
            tracks_tr_delete.append(list(tracks.keys())[track_id])

    # Delete tracks with no matches
    unmatched_tracks = set(range(len(tracks))) - set(track_indices)
    #if frame < 10:
    #    print("Unmatched tracks: ", unmatched_tracks)
    for delete_id in sorted(unmatched_tracks, reverse=True):
        tracks.pop(list(tracks.keys())[delete_id])

    #if frame < 10:
        #print("to delete: ", tracks_tr_delete)
    for delete_id in tracks_tr_delete:
        tracks.pop(delete_id)

    # Create new tracks for unmatched detections
    for j, detection_id in enumerate(range(len(detections[frame]))):
        if j not in detection_indices:
            detections[frame][detection_id][0] = track_count
            obj_id, x, y, w, h, conf, _, _, _ = detections[frame][detection_id]
            new_kalman_filter = init_kalman_filter(x, y, w, h)
            tracks[track_count] = (new_kalman_filter, [detections[frame][detection_id]])
            track_count += 1


    for detection in unmatched_det:
        detections[frame][detection][0] = track_count
        obj_id, x, y, w, h, conf, _, _, _ = detections[frame][detection_id]
        new_kalman_filter = init_kalman_filter(x, y, w, h)
        tracks[track_count] = (new_kalman_filter, [detections[frame][detection_id]])

        track_count += 1

    return tracks, track_count



def multi_object_iou_tracker(detections, sigma_iou):
    tracks = {}
    track_count = 0

    for frame in range(1, len(detections) + 1):
        tracks, track_count = match_to_track(detections, tracks, frame, track_count, sigma_iou)

    return detections


import cv2
import numpy as np

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
    output_frames_path = "output_with_kalman/"  # Replace with the desired output path
    draw_boxes_on_frames(detections, output_frames_path)

if __name__ == "__main__":
    main()

