#!/usr/bin/env python3
"""
step3_reconcile_ids.py (Verbose Edition)

This script performs ephemeral ID filtering and local spatio-temporal
reconciliation, now with extensive print statements to show the data
transformations at each step.
"""
import os
import argparse
import pickle
import json

import numpy as np
import pandas as pd

# Set pandas to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# =============================================================================
# CONFIGURATION
# =============================================================================


class ReconcileConfig:
    MIN_ID_FRAMES = 10
    MAX_ID_DISTANCE = 5
    MIN_BBOX_OVERLAP = 0.4

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_and_organize_data(raw_features_dir):
    print("="*60)
    print("STEP 1: Loading and Organizing Raw Data")
    print("="*60)
    all_files = sorted([os.path.join(raw_features_dir, f)
                       for f in os.listdir(raw_features_dir) if f.endswith('.pickle')])

    organized_data = {}
    for file_path in all_files:
        try:
            with open(file_path, 'rb') as f:
                frame_id, frame_results = pickle.load(f)
            for person in frame_results:
                tid = person.get('track_id')
                if tid is not None:
                    if tid not in organized_data:
                        organized_data[tid] = {}
                    organized_data[tid][frame_id] = person
        except (pickle.UnpicklingError, EOFError):
            print(f"[WARN] Could not read file: {file_path}")

    print(
        f"[INFO] Loaded data for {len(organized_data)} unique raw track IDs.")

    # --- VERBOSE PRINT ---
    if organized_data:
        first_id = next(iter(organized_data))
        first_frame = next(iter(organized_data[first_id]))
        print("\n[VERBOSE] Example of 'organized_data' structure:")
        print(f"  organized_data[{first_id}][{first_frame}] = {{...}}")
        print("  (A dictionary keyed by track_id, then frame_id)")
    # --- END VERBOSE ---

    return organized_data


def get_id_start_stop_df(organized_data):
    id_info = []
    for tid, frames in organized_data.items():
        if not frames:
            continue
        frame_nums = sorted(frames.keys())
        id_info.append({
            'id': tid,
            'min_frame': min(frame_nums),
            'max_frame': max(frame_nums),
            'total_frames': len(frame_nums)
        })
    df = pd.DataFrame(id_info)

    # --- VERBOSE PRINT ---
    print("\n[VERBOSE] DataFrame of Track ID Durations (`df_id_info`):")
    print(df.to_string())
    # --- END VERBOSE ---

    return df


def calculate_bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    iou = interArea / unionArea if unionArea > 0 else 0
    return iou


def resolve_chained_maps(mapping_dict):
    resolved_map = mapping_dict.copy()
    print(resolved_map)
    for k in resolved_map:
        v = resolved_map[k]
        visited = set([k])
        while v in resolved_map and v not in visited:
            visited.add(v)
            v = resolved_map[v]
        mapping_dict[k] = v
    return mapping_dict

# =============================================================================
# MAIN RECONCILIATION LOGIC
# =============================================================================


def run_local_reconciliation(features_dir, config):
    organized_data = load_and_organize_data(features_dir)
    df_id_info = get_id_start_stop_df(organized_data)

    if df_id_info.empty:
        print("[WARN] No tracking data found to reconcile.")
        return {}, None

    # 2. Filter Ephemeral IDs
    print("\n" + "="*60)
    print("STEP 2: Filtering Ephemeral IDs")
    print("="*60)
    print(f"[INFO] Filtering with MIN_ID_FRAMES = {config.MIN_ID_FRAMES}")

    persistent_ids = set(
        df_id_info[df_id_info['total_frames'] >= config.MIN_ID_FRAMES]['id'])
    ephemeral_ids = set(df_id_info.id) - persistent_ids
    df_persistent_info = df_id_info[df_id_info['id'].isin(persistent_ids)]

    # --- VERBOSE PRINT ---
    print(
        f"\n[VERBOSE] Ephemeral IDs (to be marked as -1): {sorted(list(ephemeral_ids))}")
    print(
        f"[VERBOSE] Persistent IDs (to be processed): {sorted(list(persistent_ids))}")
    print("\n[VERBOSE] DataFrame of Persistent IDs only:")
    print(df_persistent_info.to_string())
    # --- END VERBOSE ---

    print(
        f"\n  > Found {len(persistent_ids)} persistent IDs and {len(ephemeral_ids)} ephemeral IDs.")

    # 3. Local (Spatio-Temporal) Reconciliation
    print("\n" + "="*60)
    print("STEP 3: Local (Spatio-Temporal) Reconciliation")
    print("="*60)
    print(
        f"[INFO] Using MAX_ID_DISTANCE = {config.MAX_ID_DISTANCE} frames and MIN_BBOX_OVERLAP = {config.MIN_BBOX_OVERLAP}")

    local_map = {}
    for _, rowA in df_persistent_info.sort_values('max_frame').iterrows():
        idA, endA = rowA['id'], rowA['max_frame']
        print(
            f"\n[RECONCILE] Considering ID {idA} (ends at frame {endA}) for merging...")
        candidates = df_persistent_info[
            (df_persistent_info['min_frame'] > endA) &
            (df_persistent_info['min_frame'] <= endA + config.MAX_ID_DISTANCE)
        ]
        print(
            f"  [CANDIDATES] IDs starting within {config.MAX_ID_DISTANCE} frames after {endA}: {list(candidates['id'])}")
        best_iou, best_match_id = 0, None
        for _, rowB in candidates.iterrows():
            idB, startB = rowB['id'], rowB['min_frame']
            if idB in local_map:
                print(f"    [SKIP] ID {idB} already merged.")
                continue
            iou = calculate_bbox_iou(
                organized_data[idA][endA]['bbox'], organized_data[idB][startB]['bbox'])
            print(
                f"    [IOU] ID {idA} (frame {endA}) vs ID {idB} (frame {startB}): IoU={iou:.3f}")
            if iou > best_iou and iou > config.MIN_BBOX_OVERLAP:
                print(
                    f"    [MATCH] ID {idB} is new best match for ID {idA} (IoU={iou:.3f})")
                best_iou, best_match_id = iou, idB
            else:
                print(
                    f"    [NO MATCH] ID {idB} not selected (IoU={iou:.3f}, threshold={config.MIN_BBOX_OVERLAP})")

        if best_match_id:
            print(
                f"  > Found match: ID {best_match_id} will be mapped to ID {idA} (IoU: {best_iou:.3f})")
            local_map[best_match_id] = idA
            print(f"  [LOCAL_MAP UPDATE] {best_match_id} -> {idA}")
        else:
            print(f"  [NO MERGE] No suitable candidate found for ID {idA}.")

    # --- VERBOSE PRINT ---
    print("\n[VERBOSE] Initial 'local_map' of direct merges:")
    print(json.dumps(local_map, indent=4))
    # --- END VERBOSE ---

    # 4. Combine and Finalize the Map
    print("\n" + "="*60)
    print("STEP 4: Finalizing the Local ID Map")
    print("="*60)

    final_local_map = {tid: tid for tid in persistent_ids}
    final_local_map.update(local_map)

    # --- VERBOSE PRINT ---
    print("\n[VERBOSE] Map before resolving chains:")
    print(json.dumps(final_local_map, indent=4, sort_keys=True))
    # --- END VERBOSE ---

    final_local_map = resolve_chained_maps(final_local_map)

    # --- VERBOSE PRINT ---
    print("\n[VERBOSE] Map AFTER resolving chains:")
    print(json.dumps(final_local_map, indent=4, sort_keys=True))
    # --- END VERBOSE ---

    for tid in ephemeral_ids:
        final_local_map[tid] = -1

    num_final_ids = len(set(v for v in final_local_map.values() if v != -1))
    print(f"\n[SUCCESS] Local Reconciliation complete.")
    print(
        f"  > Consolidated {len(persistent_ids)} persistent IDs into {num_final_ids} locally-reconciled IDs.")

    # --- VERBOSE PRINT ---
    print("\n[VERBOSE] The final complete map (raw_id -> locally_reconciled_id):")
    # Sort for consistent display
    sorted_map = {k: final_local_map[k]
                  for k in sorted(final_local_map.keys())}
    print(json.dumps(sorted_map, indent=4))
    # --- END VERBOSE ---

    return final_local_map, organized_data

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Perform Ephemeral Filtering and Local ID Reconciliation.")
    parser.add_argument('-f', '--features', required=True,
                        help="Path to the directory containing raw feature pickle files.")
    parser.add_argument('-o', '--output', required=True,
                        help="Directory to save the output ID map.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = ReconcileConfig()

    local_id_map, _ = run_local_reconciliation(args.features, config)

    output_map_path = os.path.join(args.output, 'local_id_map.json')
    if local_id_map:
        serializable_map = {str(k): int(v) for k, v in local_id_map.items()}
        with open(output_map_path, 'w') as f:
            json.dump(serializable_map, f, indent=4, sort_keys=True)
        print(
            f"\n[FINAL SUCCESS] Locally reconciled ID map saved to: {output_map_path}")
    else:
        print("\n[INFO] No data was reconciled. No map file saved.")


if __name__ == '__main__':
    main()
