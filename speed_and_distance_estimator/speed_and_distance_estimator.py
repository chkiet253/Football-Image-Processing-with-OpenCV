import cv2
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24
    
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object_type, object_tracks in tracks.items():
            if object_type.lower() in ["ball", "referee", "referees"]:
                continue

            number_of_frames = len(object_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                track_dict_start = object_tracks[frame_num]
                track_dict_end = object_tracks[last_frame]

                if not isinstance(track_dict_start, dict) or not isinstance(track_dict_end, dict):
                    continue  # skip if not dict (e.g. Ball track)

                for track_id, _ in track_dict_start.items():
                    if track_id not in track_dict_end:
                        continue

                    start_position = track_dict_start[track_id].get('position_transformed')
                    end_position = track_dict_end[track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_mps = distance_covered / time_elapsed
                    speed_kmph = speed_mps * 3.6

                    if object_type not in total_distance:
                        total_distance[object_type] = {}

                    if track_id not in total_distance[object_type]:
                        total_distance[object_type][track_id] = 0

                    total_distance[object_type][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object_type][frame_num_batch]:
                            continue
                        tracks[object_type][frame_num_batch][track_id]['speed'] = speed_kmph
                        tracks[object_type][frame_num_batch][track_id]['distance'] = total_distance[object_type][track_id]

    
    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for object_type, object_tracks in tracks.items():
                if object_type.lower() in ["ball", "referee", "referees"]:
                    continue

                if frame_num >= len(object_tracks):
                    continue  # tránh index lỗi nếu số frame của object ít hơn

                track_data = object_tracks[frame_num]

                if not isinstance(track_data, dict):
                    continue  # đảm bảo là dict để gọi .items()

                for _, track_info in track_data.items():
                    speed = track_info.get('speed')
                    distance = track_info.get('distance')

                    if speed is None or distance is None:
                        continue

                    bbox = track_info.get('bbox')
                    if bbox is None:
                        continue

                    position = get_foot_position(bbox)
                    position = list(position)
                    position[1] += 40  # dịch vị trí xuống dưới để hiển thị text

                    position = tuple(map(int, position))
                    cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames
