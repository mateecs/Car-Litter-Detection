import os.path
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pickle
from utils import get_bbox_width, get_center_of_bbox
import sys
import cv2


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 25
        detection = []
        for i in range (0, len(frames),batch_size):
            detections_batch  =self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detection += detections_batch
        return detection

    def get_object_tracks (self, frames,read_from_stub = False ,stub_path = None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "vehicle": [],
            "waste": []
            
        }
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # Convert to supervision Detection Format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Object
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["vehicle"].append({})
            tracks["waste"].append({})


            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['vehicle']:
                    tracks["vehicle"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['waste']:
                    tracks["waste"][frame_num][track_id] = {"bbox": bbox}

    

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    
    def draw_rectangle(self, frame, bbox, color, track_id= None):
        x1, y1, x2, y2 =  map(int, bbox)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        if track_id is not None:
            cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            vehicle_dict = tracks["vehicle"][frame_num]
            waste_dict = tracks["waste"][frame_num]

            # draw vehicle
            for track_id, vehicle in vehicle_dict.items():
                frame = self.draw_rectangle(frame,vehicle["bbox"],(0,0,255), track_id)

            #Draw waste
            for track_id, waste in waste_dict.items():
                frame= self.draw_rectangle(frame,waste["bbox"],(0,255,0),'waste')

            output_video_frames.append(frame)
        return output_video_frames


