from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssginer
import cv2

def main():
    #Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('model/best.pt')
    tracks = tracker.get_object_track(video_frames,
                                      read_from_stub=True,
                                      stub_path='stubs/track_stubs.pkl')
    # # Save cropped image of a player
    # for track_id, player in tracks['Player'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     # crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # save the cropped image
    #     cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)

    #     break

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

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()