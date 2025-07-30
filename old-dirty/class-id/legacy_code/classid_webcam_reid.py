#!/usr/bin/env python3
"""
_webcam_reid.py_

A simplified, single-threaded pipeline that reads from your webcam and performs:
- Person tracking (OC-SORT)
- Pose estimation
- Face detection
- Gaze estimation
- ReID feature extraction
- Facial embedding

Results are pickled to cache/tracking_singlethread_webcam/, and you get a live display with:
- Bounding boxes
- Track IDs
- Pose keypoints
- Face boxes
- Gaze direction
"""

import os
import time
import argparse
import pickle
import cv2
import mmcv
import torch
import numpy as np
from copy import deepcopy
from mmtrack.apis import inference_mot, init_model as init_tracking_model
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from FaceWrapper import RetinaFaceInference
from GazeWrapper import GazeInference
from facenet_pytorch import InceptionResnetV1
from torchreid.utils import FeatureExtractor as REIDFeatureExtractor
from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Where your ClassID repo lives; adjust if needed
SOURCE_DIR = './'

# Model configs & weights
TRACK_CONFIG = os.path.join(SOURCE_DIR, 'configs/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half.py')
TRACK_CHECKPOINT = os.path.join(SOURCE_DIR, 'models/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth')

POSE_CONFIG = os.path.join(SOURCE_DIR, 'configs/mmlab/hrnet_w32_coco_256x192.py')
# POSE_CHECKPOINT = os.path.join(SOURCE_DIR, 'models/mmlab/hrnet_w32_coco_256x192-c78dce93_20200708.pth')
POSE_CHECKPOINT = os.path.join(SOURCE_DIR, 'models/mmlab/hrnet_w32-36af842e.pth')

DEVICE = 'cpu'   # no CUDA on MacBook Pro
TARGET_FPS = 5   # process at 5 FPS
FACE_NUM_POOL_WORKERS = 5

# Where to dump per-frame pickles
OUTPUT_DIR = 'cache/tracking_singlethread_webcam-reid'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_results(frame, frame_results):
    """Draw tracking boxes, pose keypoints, face boxes, and gaze direction"""
    if frame_results is None:
        return frame
    
    for person in frame_results:
        # Draw tracking box
        bbox = person['bbox'].astype(int)
        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {person.get('track_id', 'N/A')}", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw pose keypoints if available
        if 'keypoints' in person:
            keypoints = person['keypoints']
            for kp in keypoints:
                x, y, conf = kp
                if conf > 0.3:  # Only draw high confidence keypoints
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # Draw face box if available
        if 'face' in person and len(person['face']) > 0:
            face = person['face'][0].astype(int)
            cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), 2)
        
        # Draw gaze direction if available
        if 'gaze_2d' in person:
            gaze = person['gaze_2d']
            if isinstance(gaze, list) and len(gaze) >= 2:
                start_raw, end_raw = gaze[0], gaze[1]
                if isinstance(start_raw[0], list):  # gaze[0] = [[x, y]]
                    start_raw = start_raw[0]
                if isinstance(end_raw[0], list):
                    end_raw = end_raw[0]

                start_point = (int(start_raw[0]), int(start_raw[1]))
                end_point = (int(end_raw[0]), int(end_raw[1]))
                cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 2)

                # start_point = (int(gaze[0][0]), int(gaze[0][1]))
                # end_point = (int(gaze[1][0]), int(gaze[1][1]))
                cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 2)
        else:
            print(f"[WARN] Invalid gaze data for person: {gaze}")
    
    return frame

def main(camera_index: int):
    # -----------------------------------------------------------------------------
    # 1. Initialize all models
    # -----------------------------------------------------------------------------
    print("[INFO] Initializing models...")
    
    # Tracking model
    tracking_model = init_tracking_model(TRACK_CONFIG, TRACK_CHECKPOINT, device=DEVICE)
    
    # Pose model
    pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)
    pose_dataset = pose_model.cfg.data['test']['type']
    pose_dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    
    # Face model
    face_model = RetinaFaceInference(device=torch.device(DEVICE))
    face_threadExecutor = ThreadPoolExecutor(FACE_NUM_POOL_WORKERS)
    
    # Gaze model
    gaze_model = GazeInference(device=DEVICE)
    
    # Facial embedding model
    facial_embedding_model = InceptionResnetV1(pretrained='vggface2', device=DEVICE).eval()
    
    # ReID model
    reid_extractor = REIDFeatureExtractor(model_name='osnet_x1_0', device=DEVICE)


    # -----------------------------------------------------------------------------
    # 2. Open the webcam
    # -----------------------------------------------------------------------------
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    num_skip = int(video_fps / TARGET_FPS) if TARGET_FPS > 0 else 0
    print(f"[INFO] Camera FPS={video_fps:.1f}, processing at {TARGET_FPS} FPS")

    frame_number = 0
    last_results = None

    # -----------------------------------------------------------------------------
    # 3. Main processing loop
    # -----------------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Webcam stream ended or error reading frame.")
            break

        if num_skip == 0 or frame_number % num_skip == 0:
            t0 = time.time()
            frame_results = None    
            
            # 1. Tracking
            track_results = inference_mot(tracking_model, frame, frame_id=frame_number)
            track_bboxes = track_results['track_bboxes'][0]
            track_results = [dict(bbox=x[1:], track_id=x[0]) for x in list(track_bboxes)]

            
            if track_results:
                # 2. Pose estimation
                
                # I didn't do the normalisation that was done in `reid_added_pipeline_singlethread.py`
                
                # 2. Pose estimation
                h, w, _ = frame.shape
                config_pose = mmcv.Config.fromfile(POSE_CONFIG)
                for component in config_pose.data.test.pipeline:
                    if component['type'] == 'PoseNormalize':
                        component['mean'] = (w // 2, h // 2, .5)
                        component['max_value'] = (w, h, 1.)

                

                frame_results, _ = inference_top_down_pose_model(
                    pose_model, frame, track_results, format='xyxy',
                    dataset=pose_dataset, dataset_info=pose_dataset_info
                )
                
                # 3. Face detection
                body_frames = []
                body_indexes = []
                for body_index, person in enumerate(frame_results):
                    bbox = person['bbox'].astype(int)
                    x1, y1, x2, y2 = bbox[:4]
                    if (y2 - y1) >= 5 and (x2 - x1) >= 5 and x1 >= 0 and y1 >= 0 and x2 < frame.shape[1] and y2 < frame.shape[0]:

                        body_frame = frame[y1:y2, x1:x2, :]
                        body_frames.append(body_frame)
                        body_indexes.append(body_index)
                
                face_detections = face_threadExecutor.map(face_model.run, body_frames)
                for body_index, face_result in zip(body_indexes, face_detections):
                    frame_results[body_index]['face'] = face_result[0]
                
                # 4. Gaze estimation
                for person in frame_results:
                    if 'face' in person and len(person['face']) > 0:
                        face = person['face'][0]
                        bbox = person['bbox'].astype(int)
                        face[0] += bbox[0]  # Adjust face coordinates
                        face[1] += bbox[1]
                        face[2] += bbox[0]
                        face[3] += bbox[1]
                        
                        pred_gazes, _, points_2d, tvecs = gaze_model.run(frame, face.reshape(1, -1))
                        person.update({
                            'rvec': pred_gazes,
                            'gaze_2d': points_2d,
                            'tvec': tvecs
                        })
                
                # 5. ReID features
                body_frames = []
                body_indexes = []
                for body_index, person in enumerate(frame_results):
                    bbox = person['bbox'].astype(int)
                    x1, y1, x2, y2 = bbox[:4]
                    if (y2 - y1) >= 5 and (x2 - x1) >= 5 and x1 >= 0 and y1 >= 0 and x2 < frame.shape[1] and y2 < frame.shape[0]:

                        body_frame = frame[y1:y2, x1:x2, :]
                        body_frames.append(body_frame)
                        body_indexes.append(body_index)
                
                if body_frames:
                    body_reid_features = reid_extractor(body_frames)
                    for body_index, features in zip(body_indexes, body_reid_features):
                        frame_results[body_index]['reid_features'] = features.detach().cpu().numpy()
                
                # 6. Facial embeddings
                for person in frame_results:
                    if 'face' in person and len(person['face']) > 0:
                        face = person['face'][0].astype(int)
                        face_frame = frame[face[1]:face[3], face[0]:face[2]]
                        if face_frame.size > 0:
                            # Resize and normalize face
                            face_frame = cv2.resize(face_frame, (244, 244))
                            face_tensor = torch.from_numpy(face_frame.astype(np.float32) / 255.0)
                            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
                            with torch.no_grad():
                                embedding = facial_embedding_model(face_tensor)
                            person['face_embedding'] = embedding[0].cpu().numpy()
            
            t1 = time.time()
            print(f"[Frame {frame_number:04d}] Processing time: {t1-t0:.3f}s")
            
            # Draw results on frame
            frame = draw_results(frame, frame_results)
            cv2.imshow('ClassID Webcam Analysis', frame)
            
            # Save results
            with open(os.path.join(OUTPUT_DIR, f'{frame_number}.pickle'), 'wb') as f:
                pickle.dump((frame_number, frame_results), f)
            
            last_results = frame_results

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting on user request.")
            break

        frame_number += 1

    # -----------------------------------------------------------------------------
    # 4. Cleanup
    # -----------------------------------------------------------------------------
    with open(os.path.join(OUTPUT_DIR, 'end.pickle'), 'wb') as f:
        pickle.dump((frame_number, last_results), f)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Results saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="ClassID webcam analysis pipeline")
    p.add_argument('--camera', type=int, default=0, help="OpenCV camera index (default: 0)")
    args = p.parse_args()
    main(args.camera)
