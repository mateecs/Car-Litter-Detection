from utils import read_video, save_video
from trackers import Tracker
import cv2
from ultralytics import YOLO



def main():
    #read Video
    video_frames= read_video("input video/video 2.mp4")

    #initialize Tracker
    tracker = Tracker('models/best 4.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    #Draw Object Tracks
    out_video_frames = tracker.draw_annotations(video_frames, tracks)

    #save video
    save_video(out_video_frames, "output video/output_video_6.avi")


if __name__ == "__main__":
    main()