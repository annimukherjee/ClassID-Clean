#!/usr/bin/env python3
"""
Visualization functions for tracking data structures and ID reconciliation process
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import pickle
import os
import glob

def visualize_raw_tracking_data(tracking_data, max_ids_to_show=10):
    """
    Visualize the raw tracking data structure
    
    tracking_data format: {track_id: {frame_num: person_data}}
    """
    print("=" * 60)
    print("RAW TRACKING DATA STRUCTURE")
    print("=" * 60)
    
    print(f"Total Track IDs: {len(tracking_data)}")
    print(f"Track IDs: {list(tracking_data.keys())}")
    print()
    
    # Show structure for first few IDs
    for i, (track_id, frames) in enumerate(tracking_data.items()):
        if i >= max_ids_to_show:
            print(f"... and {len(tracking_data) - max_ids_to_show} more IDs")
            break
            
        frame_nums = sorted(frames.keys())
        min_frame, max_frame = min(frame_nums), max(frame_nums)
        
        print(f"Track ID {track_id}:")
        print(f"  ├─ Appears in {len(frames)} frames")
        print(f"  ├─ Frame range: {min_frame} → {max_frame}")
        print(f"  ├─ Frame numbers: {frame_nums[:5]}{'...' if len(frame_nums) > 5 else ''}")
        
        # Show sample person data structure
        sample_frame = frame_nums[0]
        person_data = frames[sample_frame]
        
        print(f"  └─ Sample data (frame {sample_frame}):")
        print(f"     ├─ bbox: {person_data.get('bbox', 'N/A')}")
        print(f"     ├─ track_id: {person_data.get('track_id', 'N/A')}")
        print(f"     ├─ has_face: {'face' in person_data}")
        print(f"     ├─ has_gaze: {'gaze_2d' in person_data}")
        print(f"     ├─ has_pose: {'keypoints' in person_data}")
        print(f"     ├─ has_reid: {'reid_features' in person_data}")
        print(f"     └─ has_embedding: {'face_embedding' in person_data}")
        print()

def visualize_id_timeline(tracking_data, title="ID Timeline Visualization"):
    """
    Create a timeline visualization showing when each ID is active
    """
    print("=" * 60)
    print(f"{title}")
    print("=" * 60)
    
    # Create timeline data
    timeline_data = []
    for track_id, frames in tracking_data.items():
        frame_nums = sorted(frames.keys())
        for frame_num in frame_nums:
            timeline_data.append({
                'track_id': track_id,
                'frame': frame_num,
                'active': 1
            })
    
    if not timeline_data:
        print("No data to visualize")
        return
    
    df = pd.DataFrame(timeline_data)
    
    # Create pivot table for heatmap
    pivot_df = df.pivot_table(index='track_id', columns='frame', values='active', fill_value=0)
    
    plt.figure(figsize=(15, max(6, len(tracking_data) * 0.3)))
    sns.heatmap(pivot_df, cmap='viridis', cbar_kws={'label': 'ID Active'}, 
                xticklabels=True, yticklabels=True)
    plt.title(title)
    plt.xlabel('Frame Number')
    plt.ylabel('Track ID')
    plt.tight_layout()
    plt.show()
    
    # Also print summary stats
    print(f"\nTimeline Summary:")
    for track_id, frames in tracking_data.items():
        frame_nums = sorted(frames.keys())
        gaps = []
        for i in range(1, len(frame_nums)):
            gap = frame_nums[i] - frame_nums[i-1] - 1
            if gap > 0:
                gaps.append(gap)
        
        print(f"ID {track_id}: frames {min(frame_nums)}-{max(frame_nums)} "
              f"({len(frame_nums)} total, {len(gaps)} gaps)")

def visualize_filtering_process(tracking_data, min_frames_threshold):
    """
    Visualize the ephemeral ID filtering process
    """
    print("=" * 60)
    print("ID FILTERING PROCESS")
    print("=" * 60)
    
    # Analyze frame counts
    id_stats = []
    for track_id, frames in tracking_data.items():
        frame_count = len(frames)
        frame_nums = sorted(frames.keys())
        span = max(frame_nums) - min(frame_nums) + 1
        density = frame_count / span if span > 0 else 0
        
        id_stats.append({
            'track_id': track_id,
            'frame_count': frame_count,
            'span': span,
            'density': density,
            'persistent': frame_count >= min_frames_threshold
        })
    
    df_stats = pd.DataFrame(id_stats)
    
    # Print classification
    persistent_ids = df_stats[df_stats['persistent']]['track_id'].tolist()
    ephemeral_ids = df_stats[~df_stats['persistent']]['track_id'].tolist()
    
    print(f"Threshold: {min_frames_threshold} frames")
    print(f"Persistent IDs ({len(persistent_ids)}): {persistent_ids}")
    print(f"Ephemeral IDs ({len(ephemeral_ids)}): {ephemeral_ids}")
    print()
    
    # Detailed stats
    print("Detailed ID Statistics:")
    print(df_stats.sort_values('frame_count', ascending=False).to_string(index=False))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of frame counts
    colors = ['green' if persistent else 'red' for persistent in df_stats['persistent']]
    ax1.bar(range(len(df_stats)), df_stats['frame_count'], color=colors, alpha=0.7)
    ax1.axhline(y=min_frames_threshold, color='black', linestyle='--', 
                label=f'Threshold ({min_frames_threshold})')
    ax1.set_xlabel('Track ID Index')
    ax1.set_ylabel('Frame Count')
    ax1.set_title('Frame Count per Track ID')
    ax1.legend()
    ax1.set_xticks(range(len(df_stats)))
    ax1.set_xticklabels([f"ID {id}" for id in df_stats['track_id']], rotation=45)
    
    # Histogram of frame counts
    ax2.hist(df_stats['frame_count'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=min_frames_threshold, color='red', linestyle='--', 
                label=f'Threshold ({min_frames_threshold})')
    ax2.set_xlabel('Frame Count')
    ax2.set_ylabel('Number of Track IDs')
    ax2.set_title('Distribution of Frame Counts')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return persistent_ids, ephemeral_ids

def visualize_id_reconciliation(tracking_data, id_mappings, max_distance):
    """
    Visualize the ID reconciliation process
    """
    print("=" * 60)
    print("ID RECONCILIATION PROCESS")
    print("=" * 60)
    
    if not id_mappings:
        print("No ID mappings found")
        return
    
    print("ID Mappings Found:")
    for old_id, new_id in id_mappings.items():
        print(f"  {old_id} → {new_id}")
    print()
    
    # Analyze each mapping
    print("Detailed Mapping Analysis:")
    for old_id, new_id in id_mappings.items():
        old_frames = sorted(tracking_data[old_id].keys())
        new_frames = sorted(tracking_data[new_id].keys())
        
        old_end = max(old_frames)
        new_start = min(new_frames)
        gap = new_start - old_end
        
        print(f"\nMapping: {old_id} → {new_id}")
        print(f"  Old ID frames: {min(old_frames)} to {max(old_frames)} ({len(old_frames)} frames)")
        print(f"  New ID frames: {min(new_frames)} to {max(new_frames)} ({len(new_frames)} frames)")
        print(f"  Gap between end→start: {gap} frames (max allowed: {max_distance})")
        
        # Get bounding boxes for overlap calculation
        old_bbox = tracking_data[old_id][old_end]['bbox']
        new_bbox = tracking_data[new_id][new_start]['bbox']
        
        print(f"  Old ID final bbox: {old_bbox}")
        print(f"  New ID initial bbox: {new_bbox}")

def visualize_final_mapping(complete_id_map):
    """
    Visualize the final ID mapping result
    """
    print("=" * 60)
    print("FINAL ID MAPPING RESULT")
    print("=" * 60)
    
    # Group by final ID
    mapping_groups = defaultdict(list)
    ephemeral_count = 0
    
    for old_id, new_id in complete_id_map.items():
        if new_id == -1:
            ephemeral_count += 1
        else:
            mapping_groups[new_id].append(old_id)
    
    print(f"Total original IDs: {len(complete_id_map)}")
    print(f"Final persistent IDs: {len(mapping_groups)}")
    print(f"Ephemeral IDs removed: {ephemeral_count}")
    print()
    
    # Show mapping groups
    print("Final ID Groups:")
    for new_id, old_ids in sorted(mapping_groups.items()):
        if len(old_ids) == 1:
            print(f"  Final ID {new_id}: {old_ids[0]} (no consolidation)")
        else:
            print(f"  Final ID {new_id}: {old_ids} (consolidated {len(old_ids)} IDs)")
    
    if ephemeral_count > 0:
        ephemeral_ids = [old_id for old_id, new_id in complete_id_map.items() if new_id == -1]
        print(f"  Removed IDs: {ephemeral_ids}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of ID categories
    categories = ['Persistent', 'Ephemeral']
    sizes = [len(complete_id_map) - ephemeral_count, ephemeral_count]
    colors = ['green', 'red']
    
    ax1.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('ID Classification')
    
    # Bar chart of consolidation
    consolidation_counts = [len(old_ids) for old_ids in mapping_groups.values()]
    ax2.hist(consolidation_counts, bins=max(1, max(consolidation_counts)), 
             alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('Number of Original IDs Consolidated')
    ax2.set_ylabel('Number of Final IDs')
    ax2.set_title('ID Consolidation Distribution')
    
    plt.tight_layout()
    plt.show()

def comprehensive_visualization(session_dir, min_frames=10, max_distance=150):
    """
    Run complete visualization pipeline on a session
    """
    print("🔍 COMPREHENSIVE TRACKING DATA VISUALIZATION")
    print("=" * 80)
    
    # Load data
    try:
        # Load tracking data
        tracking_data = {}
        frame_data = {}
        
        pickle_files = glob.glob(os.path.join(session_dir, "*.pickle"))
        pickle_files = [f for f in pickle_files if os.path.basename(f).split('.')[0].isdigit()]
        
        for pickle_file in pickle_files:
            frame_num = int(os.path.basename(pickle_file).split('.')[0])
            
            with open(pickle_file, 'rb') as f:
                frame_number, frame_results = pickle.load(f)
            
            if frame_results:
                for person in frame_results:
                    track_id = person.get('track_id')
                    if track_id is not None:
                        if track_id not in tracking_data:
                            tracking_data[track_id] = {}
                        tracking_data[track_id][frame_num] = person
        
        # Load ID mapping if it exists
        mapping_file = os.path.join(session_dir, 'id_mapping.pickle')
        complete_id_map = None
        if os.path.exists(mapping_file):
            with open(mapping_file, 'rb') as f:
                complete_id_map = pickle.load(f)
        
        # Run visualizations
        print(f"Session directory: {session_dir}")
        print(f"Found {len(pickle_files)} frame files")
        print()
        
        # 1. Raw data structure
        visualize_raw_tracking_data(tracking_data)
        
        # 2. Timeline visualization
        visualize_id_timeline(tracking_data, "Original ID Timeline")
        
        # 3. Filtering process
        persistent_ids, ephemeral_ids = visualize_filtering_process(tracking_data, min_frames)
        
        # 4. ID reconciliation (if mappings exist)
        if complete_id_map:
            # Extract just the reconciliation mappings
            persistent_data = {id: frames for id, frames in tracking_data.items() if id in persistent_ids}
            id_mappings = {}
            
            # Find which IDs were mapped together
            for old_id, new_id in complete_id_map.items():
                if new_id != -1 and old_id in persistent_ids:
                    # Find other IDs that map to the same new_id
                    for other_old_id, other_new_id in complete_id_map.items():
                        if (other_new_id == new_id and other_old_id != old_id and 
                            other_old_id in persistent_ids):
                            id_mappings[other_old_id] = old_id
                            break
            
            visualize_id_reconciliation(persistent_data, id_mappings, max_distance)
            
            # 5. Final mapping
            visualize_final_mapping(complete_id_map)
        
        print("\n✅ Visualization complete!")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

# Example usage function
def demo_visualization():
    """
    Demo function showing how to use the visualization functions
    """
    print("DEMO: How to use the visualization functions")
    print("=" * 50)
    
    print("""
    # To visualize your webcam session data:
    
    session_dir = 'cache/tracking_singlethread_webcam-reid'
    comprehensive_visualization(session_dir)
    
    # Or run individual visualizations:
    
    # 1. Load your data first
    tracking_data, frame_data = load_session_tracking_data(session_dir)
    
    # 2. Run specific visualizations
    visualize_raw_tracking_data(tracking_data)
    visualize_id_timeline(tracking_data)
    visualize_filtering_process(tracking_data, min_frames_threshold=10)
    
    # 3. If you have ID mappings
    id_mapping_file = os.path.join(session_dir, 'id_mapping.pickle')
    if os.path.exists(id_mapping_file):
        with open(id_mapping_file, 'rb') as f:
            complete_id_map = pickle.load(f)
        visualize_final_mapping(complete_id_map)
    """)

if __name__ == "__main__":
    demo_visualization()