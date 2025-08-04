from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
import logging
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path, batch_size=20, confidence_threshold=0.1):
        """
        Initialize the tracker with configurable parameters
        
        Args:
            model_path (str): Path to the YOLO model
            batch_size (int): Batch size for processing frames
            confidence_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        # Color scheme for different object types
        self.colors = {
            'Player': (0, 0, 255),      # Red
            'GoalKeeper': (255, 0, 0),  # Blue
            'Ball': (0, 255, 0),        # Green
            'Main Referee': (0, 255, 255),     # Yellow
            'Side Referee': (255, 0, 255),     # Magenta
            'Staff Member': (128, 128, 128)    # Gray
        }
    
    def interpolate_ball_positions(self, ball_positions):
        '''
        Interpolate missing ball positions across frames
        
        Args:
            ball_positions: List of ball detections for each frame
                        Each element is a list of ball detections in that frame
                        Format: [[], [{'bbox': [x1,y1,x2,y2]}], [], ...]
        
        Returns:
            List of interpolated ball positions with same format
        '''
        print(f"Interpolating ball positions across {len(ball_positions)} frames...")
        
        # Extract ball positions - take first ball detection in each frame if multiple exist
        extracted_positions = []
        for frame_balls in ball_positions:
            if frame_balls and len(frame_balls) > 0 and 'bbox' in frame_balls[0]:
                # Take the first ball detection if multiple balls detected
                bbox = frame_balls[0]['bbox']
                extracted_positions.append(bbox)
            else:
                # No ball detected in this frame
                extracted_positions.append([np.nan, np.nan, np.nan, np.nan])
        
        # Create DataFrame for interpolation
        df_ball_positions = pd.DataFrame(extracted_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Count missing values before interpolation
        missing_count = df_ball_positions.isna().any(axis=1).sum()
        print(f"Frames with missing ball detections: {missing_count}/{len(ball_positions)}")
        
        # Interpolate missing values using linear interpolation
        df_ball_positions = df_ball_positions.interpolate(method='linear')
        
        # Forward fill and backward fill for any remaining NaN values at edges
        df_ball_positions = df_ball_positions.bfill()  # Backward fill
        df_ball_positions = df_ball_positions.ffill()  # Forward fill
        
        # Convert back to original format
        interpolated_positions = []
        for _, row in df_ball_positions.iterrows():
            if not row.isna().any():  # If we have valid interpolated values
                bbox = [row['x1'], row['y1'], row['x2'], row['y2']]
                interpolated_positions.append([{'bbox': bbox}])
            else:
                # Still no valid position (this shouldn't happen after ffill/bfill)
                interpolated_positions.append([])
        
        # Count successful interpolations
        successful_interpolations = len([x for x in interpolated_positions if x])
        print(f"Successfully interpolated positions for {successful_interpolations}/{len(ball_positions)} frames")
        
        return interpolated_positions

    def detect_frames(self, frames):
        """
        Detect objects in frames using batch processing
        
        Args:
            frames: List of video frames
            
        Returns:
            List of detection results
        """
        detections = []
        total_batches = (len(frames) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            try:
                detections_batch = self.model.predict(
                    batch_frames, 
                    conf=self.confidence_threshold,
                    verbose=False  # Reduce output noise
                )
                detections.extend(detections_batch)
                
                # Progress logging
                current_batch = (i // self.batch_size) + 1
                if current_batch % 10 == 0 or current_batch == total_batches:
                    print(f"Processed batch {current_batch}/{total_batches}")
                    
            except Exception as e:
                logging.error(f"Error processing batch {current_batch}: {e}")
                # Add empty detection for failed batch to maintain frame alignment
                detections.extend([None] * len(batch_frames))
                
        return detections

    def get_object_track(self, frames, read_from_stub=False, stub_path=None):
        """
        Track objects across frames
        
        Args:
            frames: List of video frames
            read_from_stub (bool): Whether to read from cached results
            stub_path (str): Path to cache file
            
        Returns:
            Dictionary containing tracks for all object types
        """
        # Try to load from cache first
        if read_from_stub and stub_path and os.path.exists(stub_path):
            try:
                with open(stub_path, 'rb') as f:
                    tracks = pickle.load(f)
                print(f"Loaded tracks from cache: {stub_path}")
                return tracks
            except Exception as e:
                logging.warning(f"Failed to load cache file {stub_path}: {e}")
                print("Proceeding with fresh detection...")

        # Perform detection
        detections = self.detect_frames(frames)

        # Initialize tracking structure
        tracks = {
            'Player': [],
            'GoalKeeper': [],
            'Ball': [],
            'Main Referee': [],
            'Side Referee': [],
            'Staff Member': []
        }

        for frame_num, detection in enumerate(detections):
            # Handle failed detections
            if detection is None:
                for key in tracks:
                    if key == 'Ball':
                        tracks[key].append([])
                    else:
                        tracks[key].append({})
                continue
                
            cls_names = detection.names
            cls_name_inv = {v: k for k, v in cls_names.items()}
            
            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Track Objects (all except ball get track IDs)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize frame data
            for key in tracks:
                if key == 'Ball':
                    tracks[key].append([])
                else:
                    tracks[key].append({})
            
            # Process tracked objects (everything except ball)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Map class ID to object type and store
                for obj_type, class_name in [
                    ('Player', 'Player'),
                    ('GoalKeeper', 'GoalKeeper'), 
                    ('Main Referee', 'Main Referee'),
                    ('Side Referee', 'Side Referee'),
                    ('Staff Member', 'Staff Member')
                ]:
                    if cls_id == cls_name_inv.get(class_name, -1):
                        tracks[obj_type][frame_num][track_id] = {'bbox': bbox}
                        break
                        
            # Handle ball separately (no tracking ID, just detection)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_name_inv.get('Ball', -1):
                    tracks['Ball'][frame_num].append({'bbox': bbox})

        # Save to cache if path provided
        if stub_path:
            try:
                os.makedirs(os.path.dirname(stub_path), exist_ok=True)
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)
                print(f"Tracks saved to cache: {stub_path}")
            except Exception as e:
                logging.warning(f"Failed to save cache file {stub_path}: {e}")

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw ellipse annotation for person objects
        
        Args:
            frame: Video frame
            bbox: Bounding box coordinates
            color: RGB color tuple
            track_id: Optional track ID to display
            
        Returns:
            Annotated frame
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        # Draw ellipse at feet
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        # Draw track ID if provided
        if track_id is not None:

            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            # Draw background rectangle
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                (255, 153, 255),
                cv2.FILLED
            )
            
            # Draw track ID text
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
                
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draw triangle annotation for ball
        
        Args:
            frame: Video frame
            bbox: Bounding box coordinates
            color: RGB color tuple
            
        Returns:
            Annotated frame
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Draw all annotations on video frames
        
        Args:
            video_frames: List of video frames
            tracks: Dictionary containing all tracks
            
        Returns:
            List of annotated frames
        """
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Get frame data for all object types
            player_dict = tracks["Player"][frame_num]
            goalkeeper_dict = tracks["GoalKeeper"][frame_num]
            main_referee_dict = tracks["Main Referee"][frame_num]
            side_referee_dict = tracks["Side Referee"][frame_num]
            staff_member_dict = tracks["Staff Member"][frame_num]
            ball_list = tracks["Ball"][frame_num]

            # Draw Players with track IDs
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame, 
                    player["bbox"], 
                    color, 
                    track_id
                )
                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame, player["bbox"],(0,0,255))

            # Draw GoalKeepers with track IDs
            for track_id, goalkeeper in goalkeeper_dict.items():
                frame = self.draw_ellipse(
                    frame, 
                    goalkeeper["bbox"], 
                    self.colors['GoalKeeper'], 
                    track_id
                )

            # Draw other personnel with their respective colors
            personnel_groups = [
                (main_referee_dict, 'Main Referee'),
                (side_referee_dict, 'Side Referee'),
                (staff_member_dict, 'Staff Member')
            ]
            
            for group_dict, group_type in personnel_groups:
                for track_id, person in group_dict.items():
                    frame = self.draw_ellipse(
                        frame, 
                        person["bbox"], 
                        self.colors[group_type],
                        track_id
                    )

            # Draw Ball (green triangles)
            for ball in ball_list:
                if ball and "bbox" in ball:
                    frame = self.draw_triangle(
                        frame, 
                        ball["bbox"], 
                        self.colors['Ball']
                    )

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames

    def get_tracking_stats(self, tracks):
        """
        Get statistics about the tracking results
        
        Args:
            tracks: Dictionary containing all tracks
            
        Returns:
            Dictionary with tracking statistics
        """
        stats = {}
        
        for obj_type, track_data in tracks.items():
            if obj_type == 'Ball':
                # Count total ball detections
                total_detections = sum(len(frame_balls) for frame_balls in track_data)
                stats[obj_type] = {
                    'total_detections': total_detections,
                    'frames_with_detections': sum(1 for frame_balls in track_data if frame_balls)
                }
            else:
                # Count unique track IDs and total detections
                all_track_ids = set()
                total_detections = 0
                
                for frame_dict in track_data:
                    all_track_ids.update(frame_dict.keys())
                    total_detections += len(frame_dict)
                
                stats[obj_type] = {
                    'unique_tracks': len(all_track_ids),
                    'total_detections': total_detections,
                    'track_ids': sorted(list(all_track_ids))
                }
        
        return stats