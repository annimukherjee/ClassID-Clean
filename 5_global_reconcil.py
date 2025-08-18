import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from utils import plot_track_lifespan, calculate_bbox_iou

# -- Configuration ---
# DBSCAN parameters for cleaning facial embeddings before averaging
# eps: The max distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
DBSCAN_EPS = 0.40
DBSCAN_MIN_SAMPLES = 5 # An ID must have at least 5 high-quality face frames to get a signature

# Thresholds for matching
MIN_FACE_SIMILARITY = 0.75 # Cosine similarity; higher is more similar (max 1.0)
MIN_SPATIAL_OVERLAP = 0.50 # IoU of the average bounding boxes


def global_id_reconciliation(input_path: str, output_path: str, plot_path: str):
    """
    Performs global ID reconciliation using facial similarity and spatial proximity.
    """
    print("\n--- Step 5: Performing Global ID Reconciliation ---")
    # 1. Load data and filter for tracks with face embeddings
    print(f"[INFO] Loading feature data from: {input_path}")
    df = pd.read_pickle(input_path)
    df_faces = df.dropna(subset=['face_embedding']).copy()

    if len(df_faces['track_id'].unique()) < 2:
        print("[WARNING] Fewer than 2 track IDs with faces found. Skipping global reconciliation.")
        df.to_pickle(output_path)
        plot_track_lifespan(df, "Track Lifespans (Global Reconciliation Skipped)", plot_path)
        return

    # 2. Pre-computation: Create a "profile" for each track ID
    print("[INFO] Creating profiles (lifespan, signature embedding, avg bbox) for each track ID...")
    track_profiles = {}

    # Group by track_id once to get all necessary info
    for track_id, group in tqdm(df_faces.groupby('track_id'), desc="Creating Profiles"):
        
        # a) Create Signature Embedding using DBSCAN to find the core facial representation
        embeddings = np.vstack(group['face_embedding'].values)
        if len(embeddings) < DBSCAN_MIN_SAMPLES:
            print("Skipping DBSCAN as - Not enough data to create a reliable signature")
            continue # Not enough data to create a reliable signature
            
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(embeddings)
        # Find the largest cluster (ignoring noise, which is labeled -1)
        core_labels = clustering.labels_[clustering.labels_ != -1]
        if len(core_labels) == 0:
            print("All points were considered noise post DBSCAN")
            continue # All points were considered noise

        largest_cluster_label = pd.Series(core_labels).value_counts().idxmax()
        signature_embedding = np.median(embeddings[clustering.labels_ == largest_cluster_label], axis=0)

        # b) Calculate Lifespan
        min_frame = group['frame_id'].min()
        max_frame = group['frame_id'].max()

        # c) Calculate Average Bounding Box ("Home Base")
        avg_bbox = group[['x1', 'y1', 'x2', 'y2']].mean().values
        
        track_profiles[track_id] = {
            'signature_embedding': signature_embedding,
            'min_frame': min_frame,
            'max_frame': max_frame,
            'avg_bbox': avg_bbox
        }

    print(f"[INFO] Created {len(track_profiles)} high-quality track profiles.")

    # 3. The Main Comparison Loop: Find the best match for each track
    id_to_merge_into = {}

    track_ids = list(track_profiles.keys())
    print("[INFO] Comparing all track pairs for potential merges...")
    for i in tqdm(range(len(track_ids)), desc="Comparing Pairs"):
        for j in range(i + 1, len(track_ids)):
            id_A = track_ids[i]
            id_B = track_ids[j]
            
            # Ensure A is the earlier track for consistency
            if track_profiles[id_A]['min_frame'] > track_profiles[id_B]['min_frame']:
                id_A, id_B = id_B, id_A # Swap them

            # --- Test 1: Temporal Check ---
            if track_profiles[id_A]['max_frame'] >= track_profiles[id_B]['min_frame']:
                continue # They overlap in time, cannot be the same person

            # --- Test 2: Facial Similarity Check ---
            emb_A = track_profiles[id_A]['signature_embedding'].reshape(1, -1)
            emb_B = track_profiles[id_B]['signature_embedding'].reshape(1, -1)
            face_similarity = cosine_similarity(emb_A, emb_B)[0][0]

            if face_similarity < MIN_FACE_SIMILARITY:
                continue # Faces are not similar enough

            # --- Test 3: Spatial Proximity Check ---
            iou = calculate_bbox_iou(track_profiles[id_A]['avg_bbox'], track_profiles[id_B]['avg_bbox'])
            
            if iou < MIN_SPATIAL_OVERLAP:
                continue # They don't occupy the same general space

            # --- Success! A potential match is found ---
            # Merge the later track (B) into the earlier one (A)
            # If B is already set to be merged, we don't change it.
            # If B is not set, or this new match is better (higher similarity), we update it.
            print(f"  [POTENTIAL MATCH] ID {id_B} -> ID {id_A} (Face Sim: {face_similarity:.2f}, Spatial IoU: {iou:.2f})")
            id_to_merge_into[id_B] = id_A

    # 4. Resolve chained mappings (e.g., C->B, B->A becomes C->A)
    print("[INFO] Resolving chained merges...")
    for original_id in list(id_to_merge_into.keys()):
        target_id = id_to_merge_into[original_id]
        while target_id in id_to_merge_into:
            target_id = id_to_merge_into[target_id]
        id_to_merge_into[original_id] = target_id

    print(f"[INFO] Final global merge mapping: {id_to_merge_into}")

    # 5. Apply the mapping and save the final DataFrame
    print("[INFO] Applying merge mapping to all track IDs...")
    df['track_id'] = df['track_id'].apply(lambda x: id_to_merge_into.get(x, x))

    final_id_count = df['track_id'].nunique()
    print(f"[INFO] Global reconciliation complete. Final unique ID count: {final_id_count}")

    print(f"[INFO] Saving final reconciled data to: {output_path}")
    df.to_pickle(output_path)

    # 6. Generate the final plot
    plot_track_lifespan(
        df=df,
        title=f"Track ID Lifespans (After Global Reconciliation)\n{final_id_count} Final IDs",
        output_path=plot_path
    )
    print("--- Global ID reconciliation complete. ---")


if __name__ == '__main__':
    INPUT_PICKLE = './output/4_features_extracted.pkl'
    OUTPUT_PICKLE = './output/5_globally_reconciled.pkl'
    FINAL_PLOT = './output/5_lifespan_after_global_reconciliation.png'
    
    if not os.path.exists(INPUT_PICKLE):
        print(f"[ERROR] Input file not found: '{INPUT_PICKLE}'. Run feature extraction first.")
    else:
        start_time = time.time()
        print("hii")
        global_id_reconciliation(
            input_path=INPUT_PICKLE,
            output_path=OUTPUT_PICKLE,
            plot_path=FINAL_PLOT
        )
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")



