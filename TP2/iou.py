import cv2
import numpy as np

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


# Function to perform multi-object tracking using IoU
def match_to_track(detections, tracks, frame, track_count, sigma_iou):

    tracks_to_delete = []
    was_matched = np.zeros(len(detections[frame]))
    # For all tracks
    for track_id, track in tracks.items():
        # Get last track position
        obj_id_t, x_t, y_t, w_t, h_t, conf_t, _, _, _ = track[-1]
        last_track_box = [x_t, y_t, w_t, h_t]

        # To check if frame detection already matched
        i = 0
        matched_id = 0

        # Set max
        max_iou = float('inf')
        # For all frame detections
        for obj_id, x, y, w, h, conf, _, _, _ in detections[frame]:
            # If frame not already matched
            if not was_matched[i]:
                # Get detected box
                current_box = [x, y, w, h]

                # If computed iou between frame detection and last track detection
                # is greater than current max iou, update max
                curr_iou = calculate_iou(current_box, last_track_box)
                if max_iou == float('inf') or curr_iou > max_iou:
                    max_iou = curr_iou
                    matched_id = i

            i += 1
        # If track is unmatched
        if max_iou == float('inf'):
            # Add to list of tracks to delete
            tracks_to_delete.append(track_id)
        else:
            # Add matched frame to track
            track.append(detections[frame][matched_id])
            detections[frame][matched_id][0] = track_id

            # Update matched frames
            was_matched[matched_id] = 1


    tracks_to_delete.sort(reverse=True)
    # Delete unmatched tracks
    for delete_id in tracks_to_delete:
        tracks.pop(delete_id)

    # Create new tracks for unmatched detections
    for i in range(len(was_matched)):
        if not was_matched[i]:
            detections[frame][i][0] = track_count
            tracks[track_count] = [detections[frame][i]]
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

from matplotlib import cm

def draw_boxes_on_frames(detections, output_path):
    # Define a colormap (you can choose a different colormap if needed)
    colormap = cm.get_cmap('viridis')

    for frame_num, frame_detection in detections.items():
        # Load the frame
        frame_path = f"../ADL-Rundle-6/img1/{frame_num:06d}.jpg"  # Replace with the actual path to your frames
        frame = cv2.imread(frame_path)

        # Get unique track IDs from the current frame
        unique_track_ids = set(detection[0] for detection in frame_detection)

        # Assign a color to each unique track ID using the colormap
        track_id_to_color = {track_id: (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                             for track_id, color in zip(unique_track_ids, colormap(np.linspace(0, 1, len(unique_track_ids))))}

        # Draw bounding boxes and track IDs on the frame
        for detection in frame_detection:
            track_id, x, y, w, h, _, _, _, _ = detection
            x, y, w, h = map(int, [x, y, w, h])

            # Get the color associated with the track ID
            color = track_id_to_color[track_id]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display track ID
            cv2.putText(frame, str(int(track_id)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the frame with bounding boxes and track IDs
        output_frame_path = f"{output_path}/{frame_num:06d}_with_boxes.jpg"
        cv2.imwrite(output_frame_path, frame)


def write_detections_to_file(detections, output_file):
    with open(output_file, 'w') as file:
        for frame, frame_detections in detections.items():
            for detection in frame_detections:
                detection[5] = 1
                line = f"{frame},{','.join(map(str, detection))}\n"
                file.write(line)


# Main function
def main():
    # Load detections from the text file
    detections = load_detections("../ADL-Rundle-6/det/det.txt")

    detections = multi_object_iou_tracker(detections, 0.5)

    # for frame in range(1, len(detections) + 1):
    #     print("Frame number: ", frame)
    #     for detect in detections[frame]:
    #         print(detect[0])

    output_frames_path = "output/"  # Replace with the desired output path
    draw_boxes_on_frames(detections, output_frames_path)

    write_detections_to_file(detections, "output.txt")

if __name__ == "__main__":
    main()


