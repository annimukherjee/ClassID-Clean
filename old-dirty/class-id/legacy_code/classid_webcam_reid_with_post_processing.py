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


import pandas as pd
import glob
from sympy.geometry import Point, Polygon
import traceback

# IDs must exist for 10+ frames
MIN_ID_FRAMES = 10  # Reduced from 900 for shorter webcam sessions (30 frames/sec * 5 seconds) 

# Temporal window for matching
MAX_ID_DISTANCE = 150  # Reduced for webcam sessions

# Minimum spatial overlap required
MAX_BBOX_OVERLAP = 0.4


# Where to dump per-frame pickles
OUTPUT_DIR = 'cache/tracking_singlethread_webcam-reid'
os.makedirs(OUTPUT_DIR, exist_ok=True)



# ❌ 1. Multi-directional Matching
# CMU script checks both directions (ID A→B and B→A), yours only checks forward
# python# CMU version checks: does A end near B's start AND does B start near A's end
# # Your version only checks: does B start near A's end
# ❌ 2. Chain Mapping
# CMU script handles chains: A→B→C all become A
# python# CMU: if row['id'] in potential_id_maps.keys():
# #          potential_id_maps[successful_matched_id] = potential_id_maps[row['id']]



# -----------------------------------------------------------------------------
# POST PROCESSING FUNCTIONS FROM THE NEXT PART OF THE PIPELINE
# -----------------------------------------------------------------------------

def load_session_tracking_data(session_dir):
    """Load all pickle files from a session and build tracking matrix"""
    print(f"[INFO] Loading tracking data from {session_dir}")
    
    # Get all pickle files
    all_pickle_files = glob.glob(os.path.join(session_dir, "*.pickle"))
    
    # Filter to only include numeric frame files (exclude 'end.pickle' and 'id_mapping.pickle')
    pickle_files = []
    for pickle_file in all_pickle_files:
        filename = os.path.basename(pickle_file).split('.')[0]
        if filename.isdigit():  # Only process files with numeric names
            pickle_files.append(pickle_file)
    
    # Sort by frame number
    pickle_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    print(f"[INFO] Found {len(pickle_files)} frame files to process")
    
    tracking_data = {}
    frame_data = {}
    
    for pickle_file in pickle_files:
        frame_num = int(os.path.basename(pickle_file).split('.')[0])
        
        try:
            with open(pickle_file, 'rb') as f:
                frame_number, frame_results = pickle.load(f)
            
            frame_data[frame_num] = frame_results
            
            if frame_results:
                for person in frame_results:
                    track_id = person.get('track_id')
                    if track_id is not None:
                        if track_id not in tracking_data:
                            tracking_data[track_id] = {}
                        tracking_data[track_id][frame_num] = person
                        
        except Exception as e:
            print(f"[WARN] Error loading {pickle_file}: {e}")
            continue
    
    print(f"[INFO] Loaded {len(tracking_data)} unique track IDs across {len(frame_data)} frames")

    print(f"[DEBUG] Tracking data keys: {list(tracking_data.keys())[:10]}... (total {len(tracking_data)})")

    print(f"[DEBUG] Frame data keys: {list(frame_data.keys())[:10]}... (total {len(frame_data)})")

    print(f"[DEBUG] Loaded frame data for {len(frame_data)} frames")

    # I want to print tracking data
    if len(tracking_data) > 0:
        for track_id, frames in tracking_data.items():
            print(f"[DEBUG] Tracking data for ID {track_id}: {frames}")
            break

    return tracking_data, frame_data



def filter_ephemeral_ids(tracking_data):
    """Remove IDs that appear for less than MIN_ID_FRAMES"""
    print(f"[INFO] Filtering ephemeral IDs (threshold: {MIN_ID_FRAMES} frames)")
    
    persistent_ids = {}
    removed_ids = []
    
    for track_id, frames in tracking_data.items():
        if len(frames) >= MIN_ID_FRAMES:
            persistent_ids[track_id] = frames
        else:
            removed_ids.append(track_id)
    
    print(f"[INFO] Removed {len(removed_ids)} ephemeral IDs, kept {len(persistent_ids)} persistent IDs")
    return persistent_ids, removed_ids

def calculate_bbox_overlap(bbox1, bbox2):
    """Calculate overlap between two bounding boxes"""
    x1_tl, y1_tl, x1_br, y1_br = bbox1[:4]
    x2_tl, y2_tl, x2_br, y2_br = bbox2[:4]
    
    # Check if bounding boxes overlap
    if x1_br < x2_tl or x2_br < x1_tl or y1_br < y2_tl or y2_br < y1_tl:
        return 0.0
    
    # Calculate intersection
    x_left = max(x1_tl, x2_tl)
    y_top = max(y1_tl, y2_tl)
    x_right = min(x1_br, x2_br)
    y_bottom = min(y1_br, y2_br)
    
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Calculate union
    bbox1_area = (x1_br - x1_tl) * (y1_br - y1_tl)
    bbox2_area = (x2_br - x2_tl) * (y2_br - y2_tl)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def local_id_reconciliation(tracking_data):
    """Reconcile IDs using spatial-temporal analysis"""
    print("[INFO] Performing local ID reconciliation")
    
    # Build ID start/stop information
    id_info = []
    for track_id, frames in tracking_data.items():
        frame_nums = sorted(frames.keys())
        min_frame = min(frame_nums)
        max_frame = max(frame_nums)
        id_info.append({
            'id': track_id,
            'min_frame': min_frame,
            'max_frame': max_frame,
            'total_frames': len(frame_nums)
        })
    
    df_id_info = pd.DataFrame(id_info)
    
    # Find potential matches based on temporal proximity
    id_mappings = {}
    
    for _, row in df_id_info.iterrows():
        current_id = row['id']
        max_frame = row['max_frame']
        
        # Find IDs that start shortly after this one ends
        potential_matches = df_id_info[
            (df_id_info['min_frame'] > max_frame) & 
            (df_id_info['min_frame'] <= max_frame + MAX_ID_DISTANCE)
        ]
        
        best_match = None
        best_overlap = 0
        
        for _, match_row in potential_matches.iterrows():
            match_id = match_row['id']
            match_min_frame = match_row['min_frame']
            
            # Get bounding boxes for overlap calculation
            current_bbox = tracking_data[current_id][max_frame]['bbox']
            match_bbox = tracking_data[match_id][match_min_frame]['bbox']
            
            overlap = calculate_bbox_overlap(current_bbox, match_bbox)
            
            if overlap > MAX_BBOX_OVERLAP and overlap > best_overlap:
                best_overlap = overlap
                best_match = match_id
        
        if best_match is not None:
            id_mappings[best_match] = current_id
            print(f"[INFO] Mapping ID {best_match} -> {current_id} (overlap: {best_overlap:.3f})")
    
    return id_mappings

def reconcile_session_ids(session_dir):
    """Main function to reconcile IDs for a webcam session"""
    print(f"\n[INFO] Starting ID reconciliation for session: {session_dir}")
    
    # 1. Load tracking data
    tracking_data, frame_data = load_session_tracking_data(session_dir)
    



    if not tracking_data:
        print("[WARN] No tracking data found")
        return None
    
    print(f"[INFO] Loaded {len(tracking_data)} unique track IDs: {list(tracking_data.keys())}")
    
    # 2. Filter ephemeral IDs
    persistent_tracking_data, removed_ids = filter_ephemeral_ids(tracking_data)
    
    print(f"[DEBUG] Persistent IDs: {list(persistent_tracking_data.keys())}")
    print(f"[DEBUG] Removed ephemeral IDs: {removed_ids}")
    
    # 3. Local ID reconciliation
    id_mappings = local_id_reconciliation(persistent_tracking_data)
    
    print(f"[DEBUG] Local ID mappings: {id_mappings}")
    
    # 4. Build final ID mapping
    old_to_new_id_map = {}
    
    # Start with identity mapping for all persistent IDs
    for track_id in persistent_tracking_data.keys():
        old_to_new_id_map[track_id] = track_id
        print(f"[DEBUG] Initial mapping: {track_id} -> {track_id}")
    
    # Apply mappings (map redundant IDs to their primary ID)
    for old_id, new_id in id_mappings.items():
        old_to_new_id_map[old_id] = new_id
        print(f"[DEBUG] Reconciliation mapping: {old_id} -> {new_id}")
    
    # Map removed ephemeral IDs to a special value
    for removed_id in removed_ids:
        old_to_new_id_map[removed_id] = -1  # Special marker for removed IDs
        print(f"[DEBUG] Ephemeral mapping: {removed_id} -> -1")
    
    print(f"[DEBUG] Complete old_to_new_id_map: {old_to_new_id_map}")
    
    # 5. Reassign new sequential IDs starting from 0
    unique_persistent_ids = list(set(old_to_new_id_map[id] for id in old_to_new_id_map if old_to_new_id_map[id] != -1))
    print(f"[DEBUG] Unique persistent IDs: {unique_persistent_ids}")
    
    final_id_map = {old_id: i for i, old_id in enumerate(unique_persistent_ids)}
    print(f"[DEBUG] Final ID map: {final_id_map}")
    
    # Create complete mapping
    complete_id_map = {}
    for old_id, intermediate_id in old_to_new_id_map.items():
        if intermediate_id == -1:
            complete_id_map[old_id] = -1  # Ephemeral ID
            print(f"[DEBUG] Final ephemeral: {old_id} -> -1")
        else:
            if intermediate_id in final_id_map:
                complete_id_map[old_id] = final_id_map[intermediate_id]
                print(f"[DEBUG] Final persistent: {old_id} -> {final_id_map[intermediate_id]}")
            else:
                print(f"[ERROR] intermediate_id {intermediate_id} not found in final_id_map!")
                complete_id_map[old_id] = -1  # Fallback
    
    print(f"[DEBUG] Complete final mapping: {complete_id_map}")
    print(f"[INFO] Final mapping: {len(tracking_data)} -> {len(final_id_map)} persistent IDs")
    
    # 6. Save the mapping
    mapping_file = os.path.join(session_dir, 'id_mapping.pickle')
    with open(mapping_file, 'wb') as f:
        pickle.dump(complete_id_map, f)
    
    print(f"[INFO] ID mapping saved to {mapping_file}")
    return complete_id_map




def export_corrected_data(session_dir, id_mapping):
    """Export tracking data with corrected IDs"""
    corrected_dir = session_dir + "_corrected"
    os.makedirs(corrected_dir, exist_ok=True)
    
    pickle_files = glob.glob(os.path.join(session_dir, "*.pickle"))
    
    # Filter to only include numeric frame files (exclude 'end.pickle' and 'id_mapping.pickle')
    frame_files = []
    for pickle_file in pickle_files:
        filename = os.path.basename(pickle_file).split('.')[0]
        if filename.isdigit():  # Only process files with numeric names
            frame_files.append(pickle_file)
    
    print(f"[INFO] Processing {len(frame_files)} frame files for correction")
    
    for pickle_file in frame_files:
        frame_num = int(os.path.basename(pickle_file).split('.')[0])
        
        try:
            with open(pickle_file, 'rb') as f:
                frame_number, frame_results = pickle.load(f)
            
            if frame_results:
                corrected_results = []
                for person in frame_results:
                    old_id = person.get('track_id')
                    if old_id is not None and old_id in id_mapping:
                        new_id = id_mapping[old_id]
                        if new_id != -1:  # Skip ephemeral IDs
                            person_copy = deepcopy(person)
                            person_copy['track_id'] = new_id
                            person_copy['original_track_id'] = old_id
                            corrected_results.append(person_copy)
                
                # Save corrected data
                output_file = os.path.join(corrected_dir, os.path.basename(pickle_file))
                with open(output_file, 'wb') as f:
                    pickle.dump((frame_number, corrected_results), f)
                    
        except Exception as e:
            print(f"[WARN] Error processing {pickle_file}: {e}")
            continue
    
    print(f"[INFO] Corrected data exported to {corrected_dir}")



# -------------------------------------------------------------------------------
# END OF POST PROCESSING FUNCTIONS
# -------------------------------------------------------------------------------








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
                if conf > 0:  # Only draw high confidence keypoints
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # Draw face box if available
        if 'face' in person and len(person['face']) > 0:
            face = person['face'][0].astype(int)
            cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), 2)
        
        # DEBUG: Print gaze data structure
        print(f"[DEBUG] Person {person.get('track_id', 'N/A')} gaze data:")
        if 'gaze_2d' in person:
            print(f"  gaze_2d type: {type(person['gaze_2d'])}")
            print(f"  gaze_2d content: {person['gaze_2d']}")
        else:
            print(f"  No 'gaze_2d' key found. Available keys: {person.keys()}")
        
        # Draw gaze direction if available - FIXED VERSION
        if 'gaze_2d' in person:
            gaze = person['gaze_2d']
            print(f"  Processing gaze: {gaze}")
            
            # Handle the nested format: [[[x1, y1], [x2, y2]]]
            if isinstance(gaze, list) and len(gaze) >= 1:
                # Extract the inner list
                inner_gaze = gaze[0] if len(gaze) > 0 else None
                
                if inner_gaze and isinstance(inner_gaze, list) and len(inner_gaze) >= 2:
                    start_raw, end_raw = inner_gaze[0], inner_gaze[1]
                    print(f"  start_raw: {start_raw}, end_raw: {end_raw}")
                    
                    try:
                        start_point = (int(start_raw[0]), int(start_raw[1]))
                        end_point = (int(end_raw[0]), int(end_raw[1]))
                        print(f"  Drawing arrow from {start_point} to {end_point}")
                        
                        # Check if points are within frame bounds
                        h, w = frame.shape[:2]
                        if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                            0 <= end_point[0] < w and 0 <= end_point[1] < h):
                            cv2.arrowedLine(frame, start_point, end_point, (0, 255, 255), 3)  # Cyan and thick
                            print(f"  ✓ Arrow drawn successfully")
                        else:
                            print(f"  ✗ Points outside frame bounds (frame: {w}x{h})")
                            
                    except Exception as e:
                        print(f"  ✗ Error drawing arrow: {e}")
                        
                else:
                    print(f"  ✗ Invalid inner gaze format: {inner_gaze}")
            else:
                print(f"  ✗ Invalid outer gaze format: expected list with >=1 elements, got {type(gaze)} with length {len(gaze) if hasattr(gaze, '__len__') else 'N/A'}")
        else:
            print(f"  ✗ No gaze_2d data available")
    
    return frame


def main(camera_index: int, run_post_processing: bool = True):
    # # -----------------------------------------------------------------------------
    # # 1. Initialize all models
    # # -----------------------------------------------------------------------------
    # print("[INFO] Initializing models...")
    
    # # Tracking model
    # tracking_model = init_tracking_model(TRACK_CONFIG, TRACK_CHECKPOINT, device=DEVICE)
    
    # # Pose model
    # pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)
    # pose_dataset = pose_model.cfg.data['test']['type']
    # pose_dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    
    # # Face model
    # face_model = RetinaFaceInference(device=torch.device(DEVICE))
    # face_threadExecutor = ThreadPoolExecutor(FACE_NUM_POOL_WORKERS)
    
    # # Gaze model
    # gaze_model = GazeInference(device=DEVICE)
    
    # # Facial embedding model
    # facial_embedding_model = InceptionResnetV1(pretrained='vggface2', device=DEVICE).eval()
    
    # # ReID model
    # reid_extractor = REIDFeatureExtractor(model_name='osnet_x1_0', device=DEVICE)


    # # -----------------------------------------------------------------------------
    # # 2. Open the webcam
    # # -----------------------------------------------------------------------------
    # cap = cv2.VideoCapture(camera_index)
    # if not cap.isOpened():
    #     print(f"[ERROR] Could not open camera index {camera_index}")
    #     return

    # video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # num_skip = int(video_fps / TARGET_FPS) if TARGET_FPS > 0 else 0
    # print(f"[INFO] Camera FPS={video_fps:.1f}, processing at {TARGET_FPS} FPS")

    # frame_number = 0
    # last_results = None

    # # -----------------------------------------------------------------------------
    # # 3. Main processing loop
    # # -----------------------------------------------------------------------------
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("[INFO] Webcam stream ended or error reading frame.")
    #         break

    #     if num_skip == 0 or frame_number % num_skip == 0:
    #         t0 = time.time()
    #         frame_results = None    
            
    #         # 1. Tracking
    #         track_results = inference_mot(tracking_model, frame, frame_id=frame_number)
    #         track_bboxes = track_results['track_bboxes'][0]
    #         track_results = [dict(bbox=x[1:], track_id=x[0]) for x in list(track_bboxes)]

            
    #         if track_results:
    #             # 2. Pose estimation
                
    #             # I didn't do the normalisation that was done in `reid_added_pipeline_singlethread.py`
                




    #             frame_results, _ = inference_top_down_pose_model(
    #                 pose_model, frame, track_results, format='xyxy',
    #                 dataset=pose_dataset, dataset_info=pose_dataset_info
    #             )
                
    #             # 3. Face detection
    #             body_frames = []
    #             body_indexes = []
    #             for body_index, person in enumerate(frame_results):
    #                 bbox = person['bbox'].astype(int)
    #                 x1, y1, x2, y2 = bbox[:4]
    #                 if (y2 - y1) >= 5 and (x2 - x1) >= 5:
    #                     body_frame = frame[y1:y2, x1:x2, :]
    #                     body_frames.append(body_frame)
    #                     body_indexes.append(body_index)
                
    #             face_detections = face_threadExecutor.map(face_model.run, body_frames)
    #             for body_index, face_result in zip(body_indexes, face_detections):
    #                 frame_results[body_index]['face'] = face_result[0]
                
    #             # 4. Gaze estimation
    #             for person in frame_results:
    #                 if 'face' in person and len(person['face']) > 0:
    #                     face = person['face'][0]
    #                     bbox = person['bbox'].astype(int)
                        
    #                     # Create a copy to avoid modifying original
    #                     face_adjusted = face.copy()
    #                     face_adjusted[0] += bbox[0]  # Adjust face coordinates
    #                     face_adjusted[1] += bbox[1]
    #                     face_adjusted[2] += bbox[0]
    #                     face_adjusted[3] += bbox[1]
                        
    #                     print(f"[DEBUG] Running gaze estimation for ID {person.get('track_id', 'N/A')}")
    #                     print(f"  Face bbox: {face}")
    #                     print(f"  Body bbox: {bbox}")
    #                     print(f"  Adjusted face: {face_adjusted}")
                        
    #                     try:
    #                         pred_gazes, _, points_2d, tvecs = gaze_model.run(frame, face_adjusted.reshape(1, -1))
                            
    #                         print(f"  Gaze estimation results:")
    #                         print(f"    pred_gazes type: {type(pred_gazes)}, shape: {getattr(pred_gazes, 'shape', 'N/A')}")
    #                         print(f"    points_2d type: {type(points_2d)}, content: {points_2d}")
    #                         print(f"    tvecs type: {type(tvecs)}")
                            
    #                         person.update({
    #                             'rvec': pred_gazes,
    #                             'gaze_2d': points_2d,
    #                             'tvec': tvecs
    #                         })
                            
    #                     except Exception as e:
    #                         print(f"  ✗ Gaze estimation failed: {e}")
    #                         traceback.print_exc()
    #                 else:
    #                     print(f"[DEBUG] Skipping gaze estimation for ID {person.get('track_id', 'N/A')} - no face detected")
                                
    #             # 5. ReID features
    #             body_frames = []
    #             body_indexes = []
    #             for body_index, person in enumerate(frame_results):
    #                 bbox = person['bbox'].astype(int)
    #                 x1, y1, x2, y2 = bbox[:4]
    #                 if (y2 - y1) >= 5 and (x2 - x1) >= 5:
    #                     body_frame = frame[y1:y2, x1:x2, :]
    #                     body_frames.append(body_frame)
    #                     body_indexes.append(body_index)
                
    #             if body_frames:
    #                 body_reid_features = reid_extractor(body_frames)
    #                 for body_index, features in zip(body_indexes, body_reid_features):
    #                     frame_results[body_index]['reid_features'] = features.detach().cpu().numpy()
                
    #             # 6. Facial embeddings
    #             for person in frame_results:
    #                 if 'face' in person and len(person['face']) > 0:
    #                     face = person['face'][0].astype(int)
    #                     face_frame = frame[face[1]:face[3], face[0]:face[2]]
    #                     if face_frame.size > 0:
    #                         # Resize and normalize face
    #                         face_frame = cv2.resize(face_frame, (244, 244))
    #                         face_tensor = torch.from_numpy(face_frame.astype(np.float32) / 255.0)
    #                         face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
    #                         with torch.no_grad():
    #                             embedding = facial_embedding_model(face_tensor)
    #                         person['face_embedding'] = embedding[0].cpu().numpy()
            
    #         t1 = time.time()
    #         print(f"[Frame {frame_number:04d}] Processing time: {t1-t0:.3f}s")
            
    #         # Draw results on frame
    #         frame = draw_results(frame, frame_results)
    #         cv2.imshow('ClassID Webcam Analysis', frame)
            
    #         # Save results
    #         with open(os.path.join(OUTPUT_DIR, f'{frame_number}.pickle'), 'wb') as f:
    #             pickle.dump((frame_number, frame_results), f)
            
    #         last_results = frame_results

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         print("[INFO] Quitting on user request.")
    #         break

    #     frame_number += 1

    # # -----------------------------------------------------------------------------
    # # 4. Cleanup
    # # -----------------------------------------------------------------------------
    # with open(os.path.join(OUTPUT_DIR, 'end.pickle'), 'wb') as f:
    #     pickle.dump((frame_number, last_results), f)
    
    # cap.release()
    # cv2.destroyAllWindows()
    # print(f"[INFO] Done. Results saved to {OUTPUT_DIR}/")


    # Run post-processing ID reconciliation
    if run_post_processing:
        print("\n" + "="*50)
        print("STARTING POST-PROCESSING ID RECONCILIATION")
        print("="*50)
        
        try:
            id_mapping = reconcile_session_ids(OUTPUT_DIR)
            if id_mapping:
                print(f"[SUCCESS] ID reconciliation completed!")
                print(f"[INFO] Original IDs: {len(id_mapping)}")
                print(f"[INFO] Final persistent IDs: {len(set(v for v in id_mapping.values() if v != -1))}")
                print(f"[INFO] Ephemeral IDs removed: {sum(1 for v in id_mapping.values() if v == -1)}")
                
                # Export corrected data with new IDs
                print(f"[INFO] Exporting corrected tracking data...")
                export_corrected_data(OUTPUT_DIR, id_mapping)
                
            else:
                print("[WARN] ID reconciliation failed - no data processed")
        except Exception as e:
            print(f"[ERROR] ID reconciliation failed: {e}")
            traceback.print_exc()





if __name__ == '__main__':
    p = argparse.ArgumentParser(description="ClassID webcam analysis pipeline")
    p.add_argument('--camera', type=int, default=0, help="OpenCV camera index (default: 0)")
    p.add_argument('--no-post-processing', action='store_true', 
                    help="Skip post-processing ID reconciliation")
    args = p.parse_args()
    
    main(args.camera, run_post_processing=not args.no_post_processing)
