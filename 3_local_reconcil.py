import os
import time
import pandas as pd
from utils import plot_track_lifespan, calculate_bbox_iou
from video_utils import VideoVisualizer # ADD THIS


# --- Configuration ---
# The maximum number of frames allowed between the end of one track and the
# start of another for them to be considered for merging.
MAX_FRAME_DISTANCE = 10 # e.g., at 5 FPS, this is a 2-second window.

# The minimum Intersection over Union (IoU) required for two bounding boxes
# to be considered a spatial match.
MIN_BBOX_OVERLAP = 0.4 # 40% overlap

def local_id_reconciliation(input_path: str, output_path: str, plot_path: str):
    """
    Performs local ID reconciliation to merge fragmented tracks.

    This function identifies pairs of track IDs where one ends and another begins
    within a defined time window and spatial proximity. It merges these fragments
    to create more consistent, long-term track IDs.

    Args:
        input_path (str): Path to the filtered data from filter_ephemeral.py.
        output_path (str): Path to save the final reconciled data.
        plot_path (str): Path to save the visualization of the final lifespans.
    """
    print("\n--- Step 3: Performing Local ID Reconciliation ---")

    # 1. Load the filtered tracking data
    print(f"[INFO] Loading filtered data from: {input_path}")
    df = pd.read_pickle(input_path)
    
    if df.empty:
        print("[WARNING] Input DataFrame is empty. Nothing to reconcile.")
        df.to_pickle(output_path)
        plot_track_lifespan(df, "Track ID Lifespans (After Reconciliation - Empty)", plot_path)
        print("--- Local reconciliation step complete (input was empty). ---")
        return

    # 2. Pre-calculate the lifespan (start/end frame) for each track ID
    print("[INFO] Calculating lifespans for each track ID...")
    lifespans = df.groupby('track_id')['frame_id'].agg(['min', 'max'])
    
    id_to_merge_into = {}

    # 3. Iterate through each track ID to find potential merge candidates
    print("[INFO] Finding and scoring potential ID merges...")
    for end_id, end_row in lifespans.iterrows():
        
        # Find all tracks that start *after* the current track ends,
        # but *within* the allowed frame distance.
        potential_starts = lifespans[
            (lifespans['min'] > end_row['max']) &
            (lifespans['min'] <= end_row['max'] + MAX_FRAME_DISTANCE)
        ]
        
        if potential_starts.empty:
            continue
            
        # Get the final bounding box of the track that is ending
        end_bbox_data = df[(df['track_id'] == end_id) & (df['frame_id'] == end_row['max'])]
        # .iloc[0] is safe because frame_id/track_id is unique
        end_bbox = end_bbox_data[['x1', 'y1', 'x2', 'y2']].iloc[0].values

        best_match_id = None
        max_overlap = 0

        # For each potential candidate, calculate the spatial overlap
        for start_id, start_row in potential_starts.iterrows():
            # Get the first bounding box of the track that is starting
            start_bbox_data = df[(df['track_id'] == start_id) & (df['frame_id'] == start_row['min'])]
            start_bbox = start_bbox_data[['x1', 'y1', 'x2', 'y2']].iloc[0].values
            
            overlap = calculate_bbox_iou(end_bbox, start_bbox)
            
            # If this is the best overlap we've seen so far, store it
            if overlap > MIN_BBOX_OVERLAP and overlap > max_overlap:
                max_overlap = overlap
                best_match_id = start_id
        
        # After checking all candidates, if we found a best match, record it
        if best_match_id is not None:
            # We map the NEW id (best_match_id) to the OLD id (end_id)
            id_to_merge_into[best_match_id] = end_id
            print(f"  [MATCH] Found potential merge: {best_match_id} -> {end_id} (IoU: {max_overlap:.2f})")

    # 4. Resolve chained mappings (e.g., C->B, B->A should become C->A)
    print("[INFO] Resolving chained merges...")
    for original_id in list(id_to_merge_into.keys()):
        target_id = id_to_merge_into[original_id]
        while target_id in id_to_merge_into:
            target_id = id_to_merge_into[target_id]
        id_to_merge_into[original_id] = target_id

    print(f"[INFO] Final merge mapping: {id_to_merge_into}")
    
    # 5. Apply the mapping to the DataFrame
    print("[INFO] Applying merge mapping to track IDs...")
    # Use .get(x, x) to keep the original ID if it's not in the mapping dictionary
    df['track_id'] = df['track_id'].apply(lambda x: id_to_merge_into.get(x, x))
    
    final_id_count = df['track_id'].nunique()
    print(f"[INFO] Reconciliation complete. Final unique ID count: {final_id_count}")

    # 6. Save the reconciled DataFrame
    print(f"[INFO] Saving reconciled data to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    
    # 7. Generate the final 'after' visualization
    plot_track_lifespan(
        df=df,
        title=f"Track ID Lifespans (After Reconciliation)\n{final_id_count} Final IDs",
        output_path=plot_path
    )
    
    print("--- Local ID reconciliation complete. ---")
    
    return id_to_merge_into

if __name__ == '__main__':
    INPUT_PICKLE = './output/2_filtered_results.pkl'
    OUTPUT_PICKLE = './output/3_reconciled_results.pkl'
    FINAL_PLOT = './output/3_lifespan_after_reconciliation.png'
    # --- ADDED: Define I/O paths for the video ---
    INPUT_VIDEO = './input/video.mp4'
    OUTPUT_VIDEO = './output/video_3_local_reconciliation.mp4'
    
    if not os.path.exists(INPUT_PICKLE):
        print(f"[ERROR] Input file not found at '{INPUT_PICKLE}'.")
        print("Please run 'filter_ephemeral.py' first.")
    else:
        start_time = time.time()
        # --- MODIFIED: Capture the returned merge_map ---
        merge_map = local_id_reconciliation(
            input_path=INPUT_PICKLE,
            output_path=OUTPUT_PICKLE,
            plot_path=FINAL_PLOT
        )
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

        # --- ADDED: Generate video for this step ---
        if os.path.exists(OUTPUT_PICKLE):
            df_before = pd.read_pickle(INPUT_PICKLE)
            df_after = pd.read_pickle(OUTPUT_PICKLE)
            visualizer = VideoVisualizer(video_path=INPUT_VIDEO)
            visualizer.generate_step_3_local_reconciliation_video(df_before, df_after, merge_map, OUTPUT_VIDEO)
