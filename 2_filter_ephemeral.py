import os
import time
import pandas as pd
from utils import plot_track_lifespan

# --- Configuration ---
# *** MODIFIED: Reduced the threshold to a more sensible default for short videos. ***
# Minimum number of processed frames a track ID must appear in to be considered 'persistent'.
# The meaning of this value depends on TARGET_FPS in oc_sort.py.
# For example, with TARGET_FPS=5, a value of 5 here means the ID must exist for
# at least 1 second (5 frames / 5 fps = 1s).
MIN_ID_DURATION_FRAMES = 10

def filter_ephemeral_ids(input_path: str, output_path: str, before_plot_path: str, after_plot_path: str):
    """
    Filters short-lived ('ephemeral') track IDs from the tracking data.

    This function reads the raw tracking data, identifies track IDs that appear
    for fewer than MIN_ID_DURATION_FRAMES, removes them, and saves the
    filtered data. It also generates 'before' and 'after' plots to visualize
    the effect of the filtering.

    Args:
        input_path (str): Path to the input pickle file from oc_sort.py.
        output_path (str): Path to save the filtered data pickle file.
        before_plot_path (str): Path to save the visualization of all track IDs.
        after_plot_path (str): Path to save the visualization of persistent track IDs.
    """
    print("\n--- Step 2: Filtering Ephemeral IDs ---")

    # 1. Load the raw tracking data
    print(f"[INFO] Loading raw tracking data from: {input_path}")
    df = pd.read_pickle(input_path)
    
    if df.empty:
        print("[WARNING] Input DataFrame is empty. Nothing to filter.")
        df.to_pickle(output_path)
        plot_track_lifespan(df, "Track ID Lifespans (Before Filtering)", before_plot_path)
        plot_track_lifespan(df, "Track ID Lifespans (After Filtering - Empty)", after_plot_path)
        print("--- Ephemeral ID filtering step complete (input was empty). ---")
        return

    # 2. Generate the 'before' visualization
    plot_track_lifespan(
        df=df,
        title=f"Track ID Lifespans (Before Filtering)\n{df['track_id'].nunique()} Total IDs",
        output_path=before_plot_path
    )

    # 3. Identify ephemeral IDs by counting their occurrences
    print(f"[INFO] Identifying IDs with fewer than {MIN_ID_DURATION_FRAMES} frames...")
    # Group by 'track_id' and count the number of unique frames each appears in.
    id_counts = df.groupby('track_id')['frame_id'].nunique()
    print(id_counts)
    ephemeral_ids = id_counts[id_counts <= MIN_ID_DURATION_FRAMES].index.tolist()
    
    print(f"[INFO] Found {len(ephemeral_ids)} ephemeral IDs to remove.")

    # 4. Filter the DataFrame to keep only persistent IDs
    if ephemeral_ids:
        df_filtered = df[~df['track_id'].isin(ephemeral_ids)].copy()
        print(f"[INFO] Data filtered. Kept {df_filtered['track_id'].nunique()} persistent IDs.")
    else:
        print("[INFO] No ephemeral IDs found. No rows were removed.")
        df_filtered = df.copy()

    # 5. Save the filtered DataFrame
    print(f"[INFO] Saving filtered data to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_pickle(output_path)
    
    # 6. Generate the 'after' visualization
    plot_track_lifespan(
        df=df_filtered,
        title=f"Track ID Lifespans (After Filtering)\n{df_filtered['track_id'].nunique()} Persistent IDs",
        output_path=after_plot_path
    )

    print("--- Ephemeral ID filtering complete. ---")

if __name__ == '__main__':
    # Define input and output paths for this script
    INPUT_PICKLE = './output/1_tracking_results.pkl'
    OUTPUT_PICKLE = './output/2_filtered_results.pkl'
    BEFORE_PLOT = './output/2_1_lifespan_before_filter.png'
    AFTER_PLOT = './output/2_2_lifespan_after_filter.png'
    
    if not os.path.exists(INPUT_PICKLE):
        print(f"[ERROR] Input file not found at '{INPUT_PICKLE}'.")
        print("Please run 'oc_sort.py' first to generate the tracking data.")
    else:
        start_time = time.time()
        filter_ephemeral_ids(
            input_path=INPUT_PICKLE,
            output_path=OUTPUT_PICKLE,
            before_plot_path=BEFORE_PLOT,
            after_plot_path=AFTER_PLOT
        )
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")