import os
import time
import pandas as pd
import mmcv
import torch
import numpy as np
import cv2
from tqdm import tqdm
from video_utils import VideoVisualizer # ADD THIS

# Model initializers
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from facenet_pytorch import InceptionResnetV1
from FaceWrapper import RetinaFaceInference
from GazeWrapper import GazeInference

# --- Configuration ---
POSE_CONFIG = './configs/mmlab/hrnet_w32_coco_256x192.py'
POSE_CHECKPOINT = './models/mmlab/hrnet_w32-36af842e.pth'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
FACE_EMBEDDING_TARGET_SIZE = (244, 244)

def preprocess_face_for_embedding(face_frame, target_size):
    # ... (This function is correct and does not need changes)
    factor_0 = target_size[0] / face_frame.shape[0]
    factor_1 = target_size[1] / face_frame.shape[1]
    factor = min(factor_0, factor_1)
    dsize = (int(face_frame.shape[1] * factor), int(face_frame.shape[0] * factor))
    face_frame_resized = cv2.resize(face_frame, dsize)
    diff_0 = target_size[0] - face_frame_resized.shape[0]
    diff_1 = target_size[1] - face_frame_resized.shape[1]
    face_frame_padded = np.pad(
        face_frame_resized,
        ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
        "constant",
    )
    return face_frame_padded

def extract_features(video_path: str, input_path: str, output_path: str):
    print("\n--- Step 4: Extracting Multi-Modal Features ---")

    print(f"[INFO] Initializing models on device: '{DEVICE}'...")
    pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)
    face_model = RetinaFaceInference(device=torch.device(DEVICE))
    gaze_model = GazeInference(device=DEVICE)
    facial_embedding_model = InceptionResnetV1(pretrained='vggface2', device=DEVICE).eval()
    print("[INFO] Models initialized successfully.")

    print(f"[INFO] Loading video: {video_path}")
    video_reader = mmcv.VideoReader(video_path)
    print(f"[INFO] Loading reconciled tracking data: {input_path}")
    df = pd.read_pickle(input_path)

    if df.empty:
        print("[WARNING] Input DataFrame is empty. Nothing to process.")
        df.to_pickle(output_path)
        return

    print("[INFO] Extracting features for each detection...")
    all_processed_records = []
    
    for frame_id, group in tqdm(df.groupby('frame_id'), desc="Feature Extraction"):
        video_frame = video_reader[frame_id]
        persons_in_frame = group.to_dict('records')
        
        pose_input_list = [{'bbox': np.array([p['x1'], p['y1'], p['x2'], p['y2']])} for p in persons_in_frame]
        pose_results, _ = inference_top_down_pose_model(pose_model, video_frame, pose_input_list, format='xyxy')

        for i, person_record in enumerate(persons_in_frame):
            person_record['keypoints'] = pose_results[i]['keypoints']
            person_record.update({'face_bbox': None, 'gaze_2d': None, 'rvec': None, 'tvec': None, 'face_embedding': None})

            p_bbox = [int(c) for c in [person_record['x1'], person_record['y1'], person_record['x2'], person_record['y2']]]
            person_img = video_frame[p_bbox[1]:p_bbox[3], p_bbox[0]:p_bbox[2]]

            if person_img.size == 0:
                all_processed_records.append(person_record)
                continue

            face_bbox_relative, _ = face_model.run(person_img)

            if face_bbox_relative is not None and len(face_bbox_relative) > 0:
                face_bbox_relative = face_bbox_relative[0]
                person_record['face_bbox'] = face_bbox_relative
                
                face_bbox_absolute = face_bbox_relative.copy()
                face_bbox_absolute[0::2] += p_bbox[0]
                face_bbox_absolute[1::2] += p_bbox[1]
                
                # --- FIX: Unpack the tuple from gaze_model.run() and build a dictionary ---
                try:
                    # Gaze model returns a tuple: (rotation_vectors, _, 2d_points, translation_vectors)
                    rvecs, _, points_2d, tvecs = gaze_model.run(video_frame, face_bbox_absolute.reshape(1, -1))
                    
                    # Manually create a dictionary with the results to update the record
                    gaze_data = {
                        'rvec': rvecs,
                        'gaze_2d': points_2d,
                        'tvec': tvecs
                    }
                    person_record.update(gaze_data)
                except Exception:
                    # Gaze estimation can fail; if so, the values will remain None, which is fine.
                    pass
                # --- End of FIX ---

                face_img_cropped = video_frame[int(face_bbox_absolute[1]):int(face_bbox_absolute[3]), int(face_bbox_absolute[0]):int(face_bbox_absolute[2])]
                
                if face_img_cropped.size > 0:
                    preprocessed_face = preprocess_face_for_embedding(face_img_cropped, FACE_EMBEDDING_TARGET_SIZE)
                    face_pixels = preprocessed_face.astype(np.float32)
                    face_tensor = torch.from_numpy(face_pixels).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                    face_tensor = (face_tensor - 127.5) / 128.0

                    with torch.no_grad():
                        embedding = facial_embedding_model(face_tensor)
                    person_record['face_embedding'] = embedding[0].cpu().numpy()
            
            all_processed_records.append(person_record)

    df_features = pd.DataFrame(all_processed_records)
    print(f"\n[INFO] Saving feature-rich data to: {output_path}")
    df_features.to_pickle(output_path)

    print("--- Feature extraction complete. ---")

if __name__ == '__main__':
    VIDEO_INPUT = './input/video.mp4'
    INPUT_PICKLE = './output/3_reconciled_results.pkl'
    OUTPUT_PICKLE = './output/4_features_extracted.pkl'
    # --- ADDED: Define output path for the video ---
    OUTPUT_VIDEO = './output/video_4_feature_extraction.mp4'
    
    if not os.path.exists(INPUT_PICKLE):
        print(f"[ERROR] Input file not found: '{INPUT_PICKLE}'. Run previous steps first.")
    else:
        start_time = time.time()
        extract_features(
            video_path=VIDEO_INPUT,
            input_path=INPUT_PICKLE,
            output_path=OUTPUT_PICKLE
        )
        end_time = time.time()

        if os.path.exists(OUTPUT_PICKLE):
            print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
            print("\nVerification: Info of the saved feature DataFrame:")
            df_loaded = pd.read_pickle(OUTPUT_PICKLE)
            print(df_loaded.info())
            
            # --- ADDED: Generate video for this step ---
            visualizer = VideoVisualizer(video_path=VIDEO_INPUT)
            visualizer.generate_step_4_feature_extraction_video(df_loaded, OUTPUT_VIDEO)