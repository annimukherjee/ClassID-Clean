#!/usr/bin/env python3
"""
step4_global_reconciliation.py (Final Corrected Version)

This script performs the final stage of ID reconciliation and generates the
session-level representations for each student.
- Fixes DBSCAN sample count to be more robust for short videos.
- Fixes the pandas DataFrame creation error by correctly slicing bbox data.
"""
import os
import argparse
import pickle
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# =============================================================================
# CONFIGURATION
# =============================================================================


class GlobalConfig:
    # --- Visual Signature Generation ---
    DBSCAN_EPS = 0.4
    # --- START OF FIX 1 ---
    # Lowered min_samples to be more robust for shorter videos or IDs that
    # are not always visible. 5 samples represents ~1 second of data at 5 FPS.
    DBSCAN_MIN_SAMPLES = 5
    # --- END OF FIX 1 ---

    # --- Global Matching Thresholds ---
    MAX_EMBEDDING_DISTANCE = 0.4
    MIN_AVG_BBOX_OVERLAP = 0.3

    MIN_SESSION_COVERAGE_RATIO = 0.1

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def calculate_bbox_iou(boxA, boxB):
    interArea = max(0, min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])) * \
        max(0, min(boxA[3], boxB[3]) - max(boxA[1], boxB[1]))
    unionArea = float(((boxA[2] - boxA[0]) * (boxA[3] - boxA[1])) +
                      ((boxB[2] - boxB[0]) * (boxB[3] - boxB[1])) - interArea)
    return interArea / unionArea if unionArea > 0 else 0


def resolve_chained_maps(mapping_dict):
    while True:
        made_change = False
        for key in list(mapping_dict.keys()):
            value = mapping_dict[key]
            if value in mapping_dict and mapping_dict[key] != mapping_dict[value]:
                mapping_dict[key] = mapping_dict[value]
                made_change = True
        if not made_change:
            break
    return mapping_dict

# =============================================================================
# MAIN RECONCILIATION LOGIC
# =============================================================================


def create_visual_signatures(grouped_data, config):
    print("\n" + "="*60)
    print("STEP 2: Creating Robust Visual Signatures")
    print("="*60)
    signatures = {}
    for local_id, frames_data in grouped_data.items():
        print(
            f"\n[ID {local_id}] Gathering embeddings for DBSCAN clustering...")
        embeddings = [p['face_embedding']
                      for p in frames_data.values() if 'face_embedding' in p]
        print(f"  Found {len(embeddings)} embeddings for ID {local_id}.")
        if len(embeddings) < config.DBSCAN_MIN_SAMPLES:
            print(
                f"  > ID {local_id}: Skipping, not enough embeddings ({len(embeddings)} < {config.DBSCAN_MIN_SAMPLES})")
            continue
        print(
            f"  Running DBSCAN clustering (eps={config.DBSCAN_EPS}, min_samples={config.DBSCAN_MIN_SAMPLES})...")
        db = DBSCAN(eps=config.DBSCAN_EPS,
                    min_samples=config.DBSCAN_MIN_SAMPLES).fit(embeddings)
        unique_labels, counts = np.unique(
            db.labels_[db.labels_ != -1], return_counts=True)
        print(
            f"  DBSCAN found clusters: {dict(zip(unique_labels, counts))} (noise points: {(db.labels_ == -1).sum()})")
        if unique_labels.size == 0:
            print(
                f"  > ID {local_id}: Skipping, DBSCAN found only noise points.")
            continue
        largest_cluster_label = unique_labels[np.argmax(counts)]
        cluster_embeddings = np.array(
            embeddings)[db.labels_ == largest_cluster_label]
        print(
            f"  Largest cluster label: {largest_cluster_label} with {len(cluster_embeddings)} embeddings.")
        signatures[local_id] = np.median(cluster_embeddings, axis=0)
        print(
            f"  > ID {local_id}: Created signature from largest cluster ({len(cluster_embeddings)} embeddings).")
    print(f"\n[INFO] Created visual signatures for {len(signatures)} IDs.")
    return signatures


def run_global_reconciliation(features_dir, local_map_path, config):
    print("="*60)
    print("STEP 1: Loading Data and Applying Local Map")
    print("="*60)
    with open(local_map_path, 'r') as f:
        local_id_map = {int(k): v for k, v in json.load(f).items()}
    print(f"[DEBUG] Loaded local_id_map with {len(local_id_map)} entries.")

    all_files = sorted([os.path.join(features_dir, f)
                       for f in os.listdir(features_dir) if f.endswith('.pickle')])
    print(
        f"[DEBUG] Found {len(all_files)} raw feature files in {features_dir}.")
    grouped_data_by_local_id = {}
    for file_path in all_files:
        with open(file_path, 'rb') as f:
            frame_id, frame_results = pickle.load(f)
        print(
            f"[DEBUG] Processing frame {frame_id} from file {os.path.basename(file_path)}...")
        for person in frame_results:
            raw_tid = person.get('track_id')
            if raw_tid in local_id_map:
                local_id = local_id_map[raw_tid]
                if local_id != -1:
                    if local_id not in grouped_data_by_local_id:
                        grouped_data_by_local_id[local_id] = {}
                    if frame_id not in grouped_data_by_local_id[local_id]:
                        grouped_data_by_local_id[local_id][frame_id] = person
    print(
        f"[INFO] Grouped raw features into {len(grouped_data_by_local_id)} locally-reconciled IDs.")

    visual_signatures = create_visual_signatures(
        grouped_data_by_local_id, config)

    print("\n" + "="*60)
    print("STEP 3: Performing Global Reconciliation (Two-Factor Match)")
    print("="*60)
    id_info = []
    for local_id, frames in grouped_data_by_local_id.items():
        if not frames:
            continue
        frame_nums = sorted(frames.keys())
        id_info.append({'id': local_id, 'min_frame': min(
            frame_nums), 'max_frame': max(frame_nums)})
    df_local_id_info = pd.DataFrame(id_info).set_index('id')
    print(
        f"[DEBUG] Built DataFrame of ID start/stop frames:\n{df_local_id_info}")

    global_map = {}
    potential_ids = sorted(list(visual_signatures.keys()))
    print(f"[DEBUG] Potential IDs for global matching: {potential_ids}")

    for idA, idB in combinations(potential_ids, 2):
        print(f"\n[GLOBAL MATCH] Considering pair: ID {idA} and ID {idB}")
        # Check for temporal overlap
        if df_local_id_info.loc[idA, 'max_frame'] > df_local_id_info.loc[idB, 'min_frame']:
            print(
                f"  [SKIP] IDs {idA} and {idB} overlap in time (frames {df_local_id_info.loc[idA, 'max_frame']} > {df_local_id_info.loc[idB, 'min_frame']}).")
            continue
        dist = cdist([visual_signatures[idA]], [
                     visual_signatures[idB]], 'cosine')[0][0]
        print(f"  [VISUAL] Cosine distance between signatures: {dist:.3f}")
        if dist < config.MAX_EMBEDDING_DISTANCE:
            bboxesA = [p['bbox']
                       for p in grouped_data_by_local_id[idA].values()]
            bboxesB = [p['bbox']
                       for p in grouped_data_by_local_id[idB].values()]
            avg_bboxA = np.mean(bboxesA, axis=0)
            avg_bboxB = np.mean(bboxesB, axis=0)
            avg_iou = calculate_bbox_iou(avg_bboxA, avg_bboxB)
            print(
                f"  [SPATIAL] Average IoU between bounding boxes: {avg_iou:.3f}")
            if avg_iou > config.MIN_AVG_BBOX_OVERLAP:
                print(
                    f"  > Found GLOBAL match: ID {idB} -> ID {idA} (Dist: {dist:.3f}, Avg IoU: {avg_iou:.3f})")
                global_map[idB] = idA
            else:
                print(
                    f"  [NO MATCH] IoU {avg_iou:.3f} below threshold {config.MIN_AVG_BBOX_OVERLAP}.")
        else:
            print(
                f"  [NO MATCH] Cosine distance {dist:.3f} above threshold {config.MAX_EMBEDDING_DISTANCE}.")

    print("\n[VERBOSE] Initial 'global_map' of direct merges:\n" +
          json.dumps(global_map, indent=4))

    print("\n" + "="*60)
    print("STEP 4: Finalizing the Complete ID Map")
    print("="*60)
    final_map = {
        local_id: local_id for local_id in grouped_data_by_local_id.keys()}
    final_map.update(global_map)
    print(
        f"[DEBUG] Map before resolving chains:\n{json.dumps(final_map, indent=4)}")
    final_map = resolve_chained_maps(final_map)
    print(
        f"[DEBUG] Map after resolving chains:\n{json.dumps(final_map, indent=4)}")

    print("\n" + "="*60)
    print("STEP 5: Generating Final Session-Level Representations")
    print("="*60)
    session_reps = {}
    root_ids_to_local = {}
    for local_id, root_id in final_map.items():
        if root_id not in root_ids_to_local:
            root_ids_to_local[root_id] = []
        root_ids_to_local[root_id].append(local_id)
    print(f"[DEBUG] Grouped local IDs by root IDs: {root_ids_to_local}")

    for root_id, local_ids in root_ids_to_local.items():
        print(
            f"\n[REPRESENTATION] Building session representation for root ID {root_id} (local IDs: {local_ids})")
        if root_id not in visual_signatures:
            print(
                f"[WARN] Root ID {root_id} has no visual signature. Cannot create representation.")
            continue
        all_face_bboxes = []
        for local_id in local_ids:
            bboxes = [
                p['face'][0][:4] for p in grouped_data_by_local_id[local_id].values()
                if 'face' in p and p['face'] is not None and p['face'].size > 0
            ]
            print(
                f"  [DEBUG] Local ID {local_id} has {len(bboxes)} face bboxes.")
            all_face_bboxes.extend(bboxes)
        if not all_face_bboxes:
            print(
                f"  [WARN] No face bboxes found for root ID {root_id}. Skipping.")
            continue
        df_faces = pd.DataFrame(all_face_bboxes, columns=[
                                'x1', 'y1', 'x2', 'y2'])
        session_reps[root_id] = {
            'visual_signature': visual_signatures[root_id].tolist(),
            'positional_features': {
                'median_face_width': float((df_faces['x2'] - df_faces['x1']).median()),
                'median_face_height': float((df_faces['y2'] - df_faces['y1']).median()),
                'median_face_area': float(((df_faces['x2'] - df_faces['x1']) * (df_faces['y2'] - df_faces['y1'])).median())
            }
        }
        print(
            f"  [SUCCESS] Created session representation for root ID {root_id}.")
    print(
        f"\n[SUCCESS] Generated final representations for {len(session_reps)} students.")
    return session_reps

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Perform Global ID Reconciliation and Generate Session Representations.")
    parser.add_argument('-f', '--features', required=True,
                        help="Path to the directory containing raw feature pickle files.")
    parser.add_argument('-m', '--local-map', required=True,
                        help="Path to the local_id_map.json file from Step 3.")
    parser.add_argument('-o', '--output', required=True,
                        help="Directory to save the final session representations.")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    config = GlobalConfig()
    session_representations = run_global_reconciliation(
        args.features, args.local_map, config)
    output_path = os.path.join(args.output, 'session_representations.json')
    if session_representations:
        sorted_reps = {k: v for k, v in sorted(
            session_representations.items())}
        with open(output_path, 'w') as f:
            json.dump(sorted_reps, f, indent=4)
        print(
            f"\n[FINAL SUCCESS] Pipeline complete! Session representations saved to: {output_path}")
    else:
        print("\n[INFO] No representations were generated.")


if __name__ == '__main__':
    main()
