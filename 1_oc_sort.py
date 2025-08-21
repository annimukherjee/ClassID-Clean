import os
import time
import pandas as pd
import mmcv
from mmtrack.apis import inference_mot, init_model as init_tracking_model
from tqdm import tqdm
from video_utils import VideoVisualizer # ADD THIS

# --- Configuration ---
# Path to the MMLab config file for the OC-SORT model.
TRACK_CONFIG = './configs/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half.py'

# Path to the pre-trained model checkpoint file.
TRACK_CHECKPOINT = './models/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth'

# Device to run the model on. 'cuda:0' for GPU or 'cpu' for CPU.
DEVICE = 'cpu'

# Target FPS for processing. The script will skip frames to match this rate.
TARGET_FPS = 5

def run_oc_sort(video_path: str, output_path: str):
    """
    Runs OC-SORT tracking on a video file at a target FPS and saves the results.

    This function initializes the tracking model, processes the video by sampling
    frames to meet the TARGET_FPS, and stores the tracking results (frame ID,
    track ID, bounding box, and confidence) in a pandas DataFrame, which is then
    saved as a Pickle file.

    Args:
        video_path (str): The path to the input video file.
        output_path (str): The path where the output Pickle file will be saved.
    """
    print("--- Step 1: Running OC-SORT for Initial ID Assignment ---")

    # 1. Initialize the Tracking Model
    print(f"[INFO] Initializing tracking model from config: {TRACK_CONFIG}")
    tracking_model = init_tracking_model(TRACK_CONFIG, TRACK_CHECKPOINT, device=DEVICE)
    
    # 2. Load the video
    print(f"[INFO] Loading video: {video_path}")
    video_reader = mmcv.VideoReader(video_path)
    
    # *** ADDED: Calculate frame interval for skipping frames ***
    if TARGET_FPS <= 0:
        frame_interval = 1 # Process every frame if TARGET_FPS is 0 or negative
    else:
        # Ensure frame_interval is at least 1
        frame_interval = max(1, round(video_reader.fps / TARGET_FPS))
    
    print(f"[INFO] Video FPS: {video_reader.fps:.2f}, Target FPS: {TARGET_FPS}")
    print(f"[INFO] Processing one frame every {frame_interval} frames.")
    
    all_tracking_results = []
    processed_frame_count = 0
    
    # 3. Process video frame by frame
    print("[INFO] Processing video frames. This may take a while...")
    for frame_id, frame in enumerate(tqdm(video_reader, desc="Tracking Progress")):
        
        # *** ADDED: Skip frame if it's not the one to be processed ***
        if frame_id % frame_interval != 0:
            continue
        
        print(f"Processing frame_id: {frame_id}")
        processed_frame_count += 1
        
        # Run inference to get tracking results for the current frame
        mot_results = inference_mot(tracking_model, frame, frame_id=frame_id)
        # print(f"mot_results: {mot_results}")
        
        # The result is a dictionary containing 'track_bboxes'
        track_bboxes = mot_results['track_bboxes'][0]
        
        # Format of track_bboxes is a NumPy array of [track_id, x1, y1, x2, y2, confidence]
        for bbox in track_bboxes:
            track_id = int(bbox[0])
            x1, y1, x2, y2, confidence = bbox[1:]
            
            all_tracking_results.append({
                'frame_id': frame_id,
                'track_id': track_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'confidence': confidence
            })
            
    # 4. Convert results to a pandas DataFrame
    print(f"[INFO] Total frames in video: {len(video_reader)}. Processed frames: {processed_frame_count}.")
    if not all_tracking_results:
        print("[WARNING] No tracks were detected in the video.")
        df = pd.DataFrame(columns=['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
    else:
        print(all_tracking_results)
        print(type(all_tracking_results))
        df = pd.DataFrame(all_tracking_results)
        print(f"[INFO] Found {df['track_id'].nunique()} unique track IDs in total.")

    # 5. Save the DataFrame to a Pickle file
    print(f"[INFO] Saving tracking results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    
    print("--- OC-SORT processing complete. ---")

if __name__ == '__main__':
    
    INPUT_VIDEO = './input/video.mp4'
    OUTPUT_PICKLE = './output/1_tracking_results.pkl'
    # --- ADDED: Define output path for the video ---
    OUTPUT_VIDEO = './output/video_1_oc_sort_ids.mp4'
    
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] Input video not found at '{INPUT_VIDEO}'.")
        print("Please create an 'input' folder and place 'video.mp4' inside it.")
    else:
        start_time = time.time()
        run_oc_sort(video_path=INPUT_VIDEO, output_path=OUTPUT_PICKLE)
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

        if os.path.exists(OUTPUT_PICKLE):
            print("\nVerification: First 5 rows of the saved data:")
            df_loaded = pd.read_pickle(OUTPUT_PICKLE)
            print(df_loaded.head())
            
            # --- ADDED: Generate video for this step ---
            visualizer = VideoVisualizer(video_path=INPUT_VIDEO)
            visualizer.generate_step_1_oc_sort_video(df_loaded, OUTPUT_VIDEO)