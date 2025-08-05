from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssginer
import cv2
from player_ball_assigner import PlayerBallAssigner
import pandas as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    #Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('model/best.pt')
    tracks = tracker.get_object_track(video_frames,
                                      read_from_stub=True,
                                      stub_path='stubs/track_stubs.pkl')
    
    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate Ball Positions
    tracks["Ball"] = tracker.interpolate_ball_positions(tracks["Ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assgin Player Team
    team_assigner = TeamAssginer()
    team_assigner.assgin_team_color(video_frames[0], tracks['Player'][0])

    for frame_num, player_track in enumerate(tracks['Player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['Player'][frame_num][player_id]['team'] = team
            tracks['Player'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['Player']):
        # Get ball detections for current frame
        frame_balls = tracks['Ball'][frame_num]
        ball_assigned = False
        
        # Check if there are ball detections
        if frame_balls and len(frame_balls) > 0:
            # Take the first (most confident) ball detection
            ball_data = frame_balls[0]
            
            if 'bbox' in ball_data:
                ball_bbox = ball_data['bbox']
                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
                
                if assigned_player != -1:
                    tracks['Player'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(tracks['Player'][frame_num][assigned_player]['team'])
                    ball_assigned = True
        
        # If no ball assigned, use previous team control
        if not ball_assigned:
            if team_ball_control:  # If we have previous data
                team_ball_control.append(team_ball_control[-1])
            else:  # First frame, default to team 1
                team_ball_control.append(1)

    team_ball_control = np.array(team_ball_control)


    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    #Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()