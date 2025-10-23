#!/usr/bin/env python3
"""
YOLO Tracking Visualization with Ephemeral ID Filtering and Local ID Reconciliation
===================================================================================

This script processes a video using YOLO-11 tracking and implements the ClassID paper's
ephemeral ID filtering and local ID reconciliation techniques.

Features:
- Real-time tracking visualization with matplotlib plots
- Ephemeral ID filtering based on minimum duration threshold
- Local ID reconciliation using spatial-temporal analysis
- Color-coded bounding boxes: GREEN for persistent IDs, RED for ephemeral IDs, YELLOW for evaluating
- Configurable parameters for all filtering and reconciliation thresholds
- Enhanced video display with filtered annotations

Usage:
    python tracking_visualization_ephemeral_filtering.py --ephemeral_threshold 50 --temporal_gap_threshold 10 --min_iou_threshold 0.3
"""

import argparse
from collections import defaultdict
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


class YOLOTrackingVisualizerWithEphemeralFiltering:
    def __init__(self, model_path="../yolo11n.pt", video_path="../videos/8th-grade-vid.mp4",
                 ephemeral_threshold=50, reevaluation_interval=30,
                 temporal_gap_threshold=10, min_iou_threshold=0.3):
        """
        Initialize the YOLO tracking visualizer with ephemeral filtering and local ID reconciliation.

        Args:
            model_path (str): Path to YOLO model weights
            video_path (str): Path to input video file
            ephemeral_threshold (int): Minimum frames an ID must appear to be considered persistent
            reevaluation_interval (int): How often to re-evaluate ephemeral status (in frames)
            temporal_gap_threshold (int): Maximum frame gap for temporal matching in local reconciliation
            min_iou_threshold (float): Minimum IoU threshold for spatial matching in local reconciliation
        """
        
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.ephemeral_threshold = ephemeral_threshold
        self.reevaluation_interval = reevaluation_interval
        self.temporal_gap_threshold = temporal_gap_threshold
        self.min_iou_threshold = min_iou_threshold

        # Data storage for tracking (all IDs including ephemeral)
        self.num_of_people_in_each_frame = []
        self.frame_numbers = []
        self.all_unique_ids = set()  # All IDs (including ephemeral)
        self.all_id_history = defaultdict(list)  # All ID history
        self.current_frame = 0

        
        # Data storage for filtered tracking (persistent IDs only)
        self.filtered_num_of_people_in_each_frame = []
        self.persistent_ids = set()  # Only persistent IDs
        self.persistent_id_history = defaultdict(list)  # Only persistent ID history

        # Ephemeral filtering data
        self.ephemeral_ids = set()  # IDs that are ephemeral
        self.id_frame_counts = defaultdict(int)  # Count frames per ID

        # Video annotation data
        self.frame_detections = {}  # Store bounding boxes per frame for video annotation

        # Local ID reconciliation data
        self.id_bounding_boxes = defaultdict(list)  # Store all bounding boxes per ID
        self.id_first_frame = {}  # First frame each ID appeared
        self.id_last_frame = {}   # Last frame each ID appeared
        self.reconciliation_mapping = {}  # Maps old_id -> new_id for reconciled IDs

        # Reconciliation data for enhanced visualization
        self.reconciled_num_of_people_in_each_frame = []  # People count after reconciliation

        # Setup matplotlib for live plotting with 6 panels
        plt.ion()
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(3, 2, figsize=(16, 14))

        # Initialize plots
        self._setup_plots()

    def _setup_plots(self):
        """Setup the matplotlib subplots with enhanced reconciliation visualization."""
        # Top left: People count with reconciliation - LINE GRAPHS
        self.ax1.set_title('People Count: Raw → Ephemeral Filter → Local Reconciliation')
        self.ax1.set_xlabel('Frame Number')
        self.ax1.set_ylabel('Person Count')
        self.ax1.grid(True)

        # Top right: All ID tracking (including ephemeral)
        self.ax2.set_title('All ID Tracking Over Time (Color: Red=Ephemeral, Green=Persistent, Purple=Reconciled)')
        self.ax2.set_xlabel('Frame Number')
        self.ax2.set_ylabel('Person ID')
        self.ax2.grid(True)

        # Middle left: Processing stages comparison - LINE GRAPHS
        self.ax3.set_title('Processing Pipeline: Raw → Ephemeral Filter → Local Reconciliation')
        self.ax3.set_xlabel('Frame Number')
        self.ax3.set_ylabel('Person Count')
        self.ax3.grid(True)

        # Middle right: Local reconciliation mappings
        self.ax4.set_title('Local Reconciliation Mappings (Purple=Reconciled Pairs)')
        self.ax4.set_xlabel('Frame Number')
        self.ax4.set_ylabel('Person ID')
        self.ax4.grid(True)

        # Bottom left: DEDICATED RECONCILIATION TEMPORAL VIEW (like paper figure)
        self.ax5.set_title('🔄 LOCAL RECONCILIATION: Temporal View with Arrows (ClassID Paper Style)')
        self.ax5.set_xlabel('Time (Frame Number)')
        self.ax5.set_ylabel('ID Assignment (Multi-Object Tracking)')
        self.ax5.grid(True, alpha=0.3)

        # Bottom right: Reconciliation statistics
        self.ax6.set_title('📊 Reconciliation Statistics & Impact')
        self.ax6.set_xlabel('Processing Stage')
        self.ax6.set_ylabel('Count')
        self.ax6.grid(True)

        plt.tight_layout()

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (tuple): (x1, y1, x2, y2) format
            box2 (tuple): (x1, y1, x2, y2) format

        Returns:
            float: IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection coordinates
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        # Calculate intersection area
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _get_representative_bbox(self, track_id):
        """
        Get representative bounding box for an ID (median of all bboxes).

        Args:
            track_id (int): The tracking ID

        Returns:
            tuple: Representative bounding box (x1, y1, x2, y2)
        """
        if track_id not in self.id_bounding_boxes or not self.id_bounding_boxes[track_id]:
            return None

        boxes = self.id_bounding_boxes[track_id]

        # Calculate median coordinates
        x1_coords = [box[0] for box in boxes]
        y1_coords = [box[1] for box in boxes]
        x2_coords = [box[2] for box in boxes]
        y2_coords = [box[3] for box in boxes]

        x1_median = sorted(x1_coords)[len(x1_coords) // 2]
        y1_median = sorted(y1_coords)[len(y1_coords) // 2]
        x2_median = sorted(x2_coords)[len(x2_coords) // 2]
        y2_median = sorted(y2_coords)[len(y2_coords) // 2]

        return (x1_median, y1_median, x2_median, y2_median)

    def _perform_local_reconciliation(self):
        """
        Perform local ID reconciliation based on spatial-temporal analysis.
        This matches IDs that are temporally non-overlapping but spatially similar.
        """
        print(f"[LOCAL_RECONCILIATION] Starting with {len(self.persistent_ids)} persistent IDs")

        # Reset reconciliation mapping
        self.reconciliation_mapping = {}

        # Get all persistent IDs sorted by first appearance
        persistent_ids_list = sorted(list(self.persistent_ids),
                                   key=lambda x: self.id_first_frame.get(x, float('inf')))

        matched_pairs = []

        # Find potential matches based on temporal proximity
        for i, id1 in enumerate(persistent_ids_list):
            if id1 in self.reconciliation_mapping:  # Already matched
                continue

            id1_last = self.id_last_frame.get(id1, 0)
            id1_bbox = self._get_representative_bbox(id1)

            if id1_bbox is None:
                continue

            best_match = None
            best_iou = 0.0

            # Look for IDs that start shortly after this ID ends
            for j, id2 in enumerate(persistent_ids_list[i+1:], i+1):
                if id2 in self.reconciliation_mapping:  # Already matched
                    continue

                id2_first = self.id_first_frame.get(id2, float('inf'))

                # Check temporal proximity (id2 should start within threshold frames after id1 ends)
                temporal_gap = id2_first - id1_last
                if temporal_gap <= 0 or temporal_gap > self.temporal_gap_threshold:
                    continue

                # Check spatial overlap
                id2_bbox = self._get_representative_bbox(id2)
                if id2_bbox is None:
                    continue

                iou = self._calculate_iou(id1_bbox, id2_bbox)

                # Check if this is the best match so far
                if iou >= self.min_iou_threshold and iou > best_iou:
                    best_match = id2
                    best_iou = iou

            # Record the best match if found
            if best_match is not None:
                matched_pairs.append((id1, best_match, best_iou))
                self.reconciliation_mapping[best_match] = id1
                print(f"[LOCAL_RECONCILIATION] Matched ID {best_match} -> ID {id1} (IoU: {best_iou:.3f}, Gap: {self.id_first_frame[best_match] - id1_last} frames)")

        # Apply reconciliation mapping to tracking data
        if matched_pairs:
            self._apply_reconciliation_mapping()
            # Rebuild reconciled tracking data
            self._build_reconciled_tracking_data()

        print(f"[LOCAL_RECONCILIATION] Completed. {len(matched_pairs)} ID pairs reconciled")
        print(f"[LOCAL_RECONCILIATION] Remaining persistent IDs: {len(self.persistent_ids)}")

    def _apply_reconciliation_mapping(self):
        """
        Apply the reconciliation mapping to all tracking data structures.
        """
        if not self.reconciliation_mapping:
            return

        # Update all_id_history by merging reconciled IDs
        new_all_id_history = defaultdict(list)
        for track_id, frames in self.all_id_history.items():
            if track_id in self.reconciliation_mapping:
                # This ID should be merged with another
                target_id = self.reconciliation_mapping[track_id]
                new_all_id_history[target_id].extend(frames)
            else:
                new_all_id_history[track_id] = frames

        # Sort frame lists and remove duplicates
        for track_id in new_all_id_history:
            new_all_id_history[track_id] = sorted(list(set(new_all_id_history[track_id])))

        self.all_id_history = new_all_id_history

        # Update id_frame_counts
        new_id_frame_counts = defaultdict(int)
        for track_id, frames in self.all_id_history.items():
            new_id_frame_counts[track_id] = len(frames)
        self.id_frame_counts = new_id_frame_counts

        # Update persistent_ids by removing reconciled IDs
        ids_to_remove = set(self.reconciliation_mapping.keys())
        self.persistent_ids = self.persistent_ids - ids_to_remove
        self.all_unique_ids = self.all_unique_ids - ids_to_remove

        # Update id_first_frame and id_last_frame
        for track_id, frames in self.all_id_history.items():
            if frames:
                self.id_first_frame[track_id] = min(frames)
                self.id_last_frame[track_id] = max(frames)

        # Remove data for reconciled IDs
        for reconciled_id in ids_to_remove:
            self.id_first_frame.pop(reconciled_id, None)
            self.id_last_frame.pop(reconciled_id, None)
            self.id_bounding_boxes.pop(reconciled_id, None)

        # Update persistent_id_history
        self._build_persistent_tracking_data()

    def _build_reconciled_tracking_data(self):
        """
        Build tracking data after local reconciliation.
        """
        # Rebuild reconciled frame counts for all frames processed so far
        self.reconciled_num_of_people_in_each_frame = []
        for frame_num in self.frame_numbers:
            reconciled_count = sum(
                1 for track_id, frames in self.persistent_id_history.items()
                if frame_num in frames
            )
            self.reconciled_num_of_people_in_each_frame.append(reconciled_count)

    def _identify_ephemeral_ids(self):
        """Identify which IDs are ephemeral based on the threshold."""
        self.ephemeral_ids = {
            track_id for track_id, count in self.id_frame_counts.items()
            if count < self.ephemeral_threshold
        }
        self.persistent_ids = self.all_unique_ids - self.ephemeral_ids

        print(f"[FILTER] Total unique IDs: {len(self.all_unique_ids)}")
        print(f"[FILTER] Ephemeral IDs (< {self.ephemeral_threshold} frames): {len(self.ephemeral_ids)}")
        print(f"[FILTER] Persistent IDs (≥ {self.ephemeral_threshold} frames): {len(self.persistent_ids)}")

        # Build persistent tracking data
        self._build_persistent_tracking_data()

        # Perform local ID reconciliation on persistent IDs
        if len(self.persistent_ids) > 1:  # Only reconcile if we have multiple persistent IDs
            self._perform_local_reconciliation()

    def _build_persistent_tracking_data(self):
        """Build tracking data for persistent IDs only."""
        self.persistent_id_history = {
            track_id: frames for track_id, frames in self.all_id_history.items()
            if track_id in self.persistent_ids
        }

        # Rebuild filtered frame counts for all frames processed so far
        self.filtered_num_of_people_in_each_frame = []
        for frame_num in self.frame_numbers:
            persistent_count = sum(
                1 for track_id, frames in self.persistent_id_history.items()
                if frame_num in frames
            )
            self.filtered_num_of_people_in_each_frame.append(persistent_count)

    def _get_box_color_and_label(self, track_id: int) -> Tuple[Tuple[int, int, int], str]:
        """
        Get color and label for bounding box based on whether ID is ephemeral, persistent, or reconciled.

        Args:
            track_id (int): The tracking ID

        Returns:
            Tuple[Tuple[int, int, int], str]: (BGR_color, label_text)
        """
        # Check if this ID was reconciled (merged into another ID)
        if track_id in self.reconciliation_mapping:
            target_id = self.reconciliation_mapping[track_id]
            return (255, 0, 255), f"ID: {track_id}→{target_id} (Reconciled)"  # Magenta for reconciled
        elif track_id in self.ephemeral_ids:
            return (0, 0, 255), f"ID: {track_id} (Removed)"  # Red for ephemeral
        elif track_id in self.persistent_ids:
            return (0, 255, 0), f"ID: {track_id}"  # Green for persistent
        else:
            # ID hasn't been classified yet (still being evaluated)
            return (0, 255, 255), f"ID: {track_id} (Evaluating)"  # Yellow for evaluating

    def _draw_text_with_background(self, frame, text: str, pos: Tuple[int, int],
                                 font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """Draw text with background for better visibility."""
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        start_point = (pos[0], pos[1] - text_height - 5)
        end_point = (pos[0] + text_width + 5, pos[1])
        cv2.rectangle(frame, start_point, end_point, bg_color, -1)
        cv2.putText(frame, text, (pos[0], pos[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)

    def update_plots(self):
        """Update all matplotlib plots with enhanced reconciliation visualization and arrows."""
        if len(self.frame_numbers) > 0:
            # Top left: People count comparison - LINE GRAPHS
            self.ax1.clear()

            # Plot raw count
            self.ax1.plot(self.frame_numbers, self.num_of_people_in_each_frame,
                         'b-', linewidth=2, alpha=0.7, label='Raw Count')

            # Plot filtered count if available
            if self.filtered_num_of_people_in_each_frame and len(self.filtered_num_of_people_in_each_frame) == len(self.frame_numbers):
                self.ax1.plot(self.frame_numbers, self.filtered_num_of_people_in_each_frame,
                             'g-', linewidth=2, alpha=0.8, label='After Ephemeral Filter')

            # Plot reconciled count if available
            if self.reconciled_num_of_people_in_each_frame and len(self.reconciled_num_of_people_in_each_frame) == len(self.frame_numbers):
                self.ax1.plot(self.frame_numbers, self.reconciled_num_of_people_in_each_frame,
                             'purple', linewidth=3, alpha=0.9, label='After Local Reconciliation')

            current_raw = self.num_of_people_in_each_frame[-1] if self.num_of_people_in_each_frame else 0
            current_filtered = self.filtered_num_of_people_in_each_frame[-1] if self.filtered_num_of_people_in_each_frame else 0
            current_reconciled = self.reconciled_num_of_people_in_each_frame[-1] if self.reconciled_num_of_people_in_each_frame else 0

            self.ax1.set_title(f'People Count: Raw={current_raw}, Filtered={current_filtered}, Reconciled={current_reconciled}')
            self.ax1.set_xlabel('Frame Number')
            self.ax1.set_ylabel('Person Count')
            self.ax1.grid(True)
            self.ax1.legend()

            # Top right: All ID tracking with enhanced reconciliation display
            self.ax2.clear()
            if self.all_unique_ids:
                for track_id, frames in self.all_id_history.items():
                    if frames:
                        # Determine color and style based on status
                        if track_id in self.reconciliation_mapping:
                            color = 'purple'
                            alpha = 0.9
                            marker = 's'  # Square for reconciled
                            size = 25
                        elif track_id in self.ephemeral_ids:
                            color = 'red'
                            alpha = 0.4
                            marker = 'x'  # X for ephemeral
                            size = 15
                        elif track_id in self.persistent_ids:
                            color = 'green'
                            alpha = 0.8
                            marker = 'o'  # Circle for persistent
                            size = 20
                        else:
                            color = 'orange'
                            alpha = 0.6
                            marker = '^'  # Triangle for evaluating
                            size = 18

                        self.ax2.scatter(frames, [track_id] * len(frames),
                                       c=color, s=size, alpha=alpha, marker=marker)

            reconciled_count = len(self.reconciliation_mapping)
            self.ax2.set_title(f'ID Tracking: Total={len(self.all_unique_ids)}, Ephemeral={len(self.ephemeral_ids)}, Reconciled={reconciled_count}')
            self.ax2.set_xlabel('Frame Number')
            self.ax2.set_ylabel('Person ID')
            self.ax2.grid(True)

            # Middle left: Processing stages comparison - LINE GRAPHS
            self.ax3.clear()

            if len(self.frame_numbers) > 0:
                # Always show raw count
                self.ax3.plot(self.frame_numbers, self.num_of_people_in_each_frame,
                             'b-', linewidth=1.5, alpha=0.6, label='1. Raw Detection')

                # Show ephemeral filtered if available
                if self.filtered_num_of_people_in_each_frame and len(self.filtered_num_of_people_in_each_frame) == len(self.frame_numbers):
                    self.ax3.plot(self.frame_numbers, self.filtered_num_of_people_in_each_frame,
                                 'g-', linewidth=2, alpha=0.7, label='2. Ephemeral Filtered')

                # Show reconciled if available (MOST PROMINENT)
                if self.reconciled_num_of_people_in_each_frame and len(self.reconciled_num_of_people_in_each_frame) == len(self.frame_numbers):
                    self.ax3.plot(self.frame_numbers, self.reconciled_num_of_people_in_each_frame,
                                 'purple', linewidth=3, alpha=1.0, label='3. Local Reconciliation (FINAL)')

            self.ax3.set_title('Processing Pipeline: Raw → Ephemeral Filter → Local Reconciliation')
            self.ax3.set_xlabel('Frame Number')
            self.ax3.set_ylabel('Person Count')
            self.ax3.grid(True)
            self.ax3.legend()

            # Middle right: Reconciliation mappings display
            self.ax4.clear()

            if self.reconciliation_mapping:
                # Show reconciliation mappings prominently
                for source_id, target_id in self.reconciliation_mapping.items():
                    if source_id in self.all_id_history and target_id in self.all_id_history:
                        source_frames = self.all_id_history[source_id]
                        target_frames = self.all_id_history[target_id]

                        # Show source in red (removed)
                        self.ax4.scatter(source_frames, [source_id] * len(source_frames),
                                       c='red', s=30, alpha=0.6, marker='x')

                        # Show target in purple (merged into)
                        combined_frames = sorted(set(source_frames + target_frames))
                        self.ax4.scatter(combined_frames, [target_id] * len(combined_frames),
                                       c='purple', s=40, alpha=0.9, marker='o')

                        # Draw connection line
                        if source_frames and target_frames:
                            last_source_frame = max(source_frames)
                            first_target_frame = min(target_frames)
                            self.ax4.plot([last_source_frame, first_target_frame],
                                        [source_id, target_id], 'purple', linewidth=3, alpha=0.7)

                self.ax4.set_title(f'Local Reconciliation Results: {len(self.reconciliation_mapping)} pairs merged')
            else:
                # Show persistent IDs if no reconciliation yet
                if self.persistent_id_history:
                    for track_id, frames in self.persistent_id_history.items():
                        self.ax4.scatter(frames, [track_id] * len(frames),
                                       c='green', s=20, alpha=0.8, marker='o')

                self.ax4.set_title('Local Reconciliation: No reconciliation performed yet')

            self.ax4.set_xlabel('Frame Number')
            self.ax4.set_ylabel('Person ID')
            self.ax4.grid(True)

            # Bottom left: DEDICATED RECONCILIATION TEMPORAL VIEW with ARROWS (like paper figure)
            self._update_reconciliation_temporal_view()

            # Bottom right: Reconciliation statistics
            self._update_reconciliation_statistics()

            plt.tight_layout()
            plt.pause(0.01)

    def _update_reconciliation_temporal_view(self):
        """
        Update the dedicated reconciliation temporal view with arrows (like ClassID paper figure).
        This shows the temporal patterns of ID assignments with clear reconciliation arrows.
        """
        self.ax5.clear()
        self.ax5.set_title('🔄 LOCAL RECONCILIATION: Temporal View with Arrows (ClassID Paper Style)')
        self.ax5.set_xlabel('Time (Frame Number)')
        self.ax5.set_ylabel('ID Assignment (Multi-Object Tracking)')
        self.ax5.grid(True, alpha=0.3)

        if not self.persistent_id_history and not self.reconciliation_mapping:
            self.ax5.text(0.5, 0.5, 'No reconciliation data yet',
                         transform=self.ax5.transAxes, ha='center', va='center',
                         fontsize=12, alpha=0.7)
            return

        # Get all relevant IDs for display
        all_ids_to_show = set()
        if self.persistent_id_history:
            all_ids_to_show.update(self.persistent_id_history.keys())
        if self.reconciliation_mapping:
            all_ids_to_show.update(self.reconciliation_mapping.keys())
            all_ids_to_show.update(self.reconciliation_mapping.values())

        # Sort IDs by first appearance for better visualization
        sorted_ids = sorted(list(all_ids_to_show),
                           key=lambda x: self.id_first_frame.get(x, float('inf')))

        y_positions = {id_val: i for i, id_val in enumerate(sorted_ids)}

        # Plot all persistent ID timelines first
        for track_id in sorted_ids:
            if track_id in self.persistent_id_history:
                frames = self.persistent_id_history[track_id]
                y_pos = y_positions[track_id]

                # Determine color and style
                if track_id in [target for target in self.reconciliation_mapping.values()]:
                    # This ID is a target of reconciliation (receives merged data)
                    color = 'purple'
                    alpha = 0.9
                    linewidth = 4
                elif track_id in self.reconciliation_mapping:
                    # This ID was reconciled away (should not appear in persistent history anymore)
                    continue  # Skip reconciled-away IDs
                else:
                    # Regular persistent ID
                    color = 'green'
                    alpha = 0.7
                    linewidth = 3

                # Draw timeline as horizontal line segments
                if len(frames) > 1:
                    # Draw as continuous line
                    self.ax5.plot([min(frames), max(frames)], [y_pos, y_pos],
                                 color=color, linewidth=linewidth, alpha=alpha,
                                 solid_capstyle='round')
                else:
                    # Single frame - draw as point
                    self.ax5.scatter([frames[0]], [y_pos], c=color, s=50, alpha=alpha)

                # Add ID label at the start
                self.ax5.text(min(frames) - 5, y_pos, f'{track_id}',
                             va='center', ha='right', fontweight='bold',
                             color=color, fontsize=10)

        # Draw reconciliation arrows (the key feature!)
        if self.reconciliation_mapping:
            for source_id, target_id in self.reconciliation_mapping.items():
                if source_id in self.all_id_history and target_id in y_positions:
                    source_frames = self.all_id_history.get(source_id, [])
                    target_frames = self.all_id_history.get(target_id, [])

                    if source_frames and target_frames:
                        # Source timeline (ephemeral, in red/orange)
                        source_y = len(sorted_ids)  # Place reconciled IDs above
                        source_last = max(source_frames)
                        source_first = min(source_frames)

                        # Draw source timeline in orange (ephemeral that got reconciled)
                        self.ax5.plot([source_first, source_last], [source_y, source_y],
                                     color='orange', linewidth=2, alpha=0.8, linestyle='--')
                        self.ax5.text(source_first - 5, source_y, f'{source_id}',
                                     va='center', ha='right', fontweight='bold',
                                     color='orange', fontsize=10)

                        # Target position
                        target_y = y_positions[target_id]
                        target_first = min(target_frames)

                        # Draw RECONCILIATION ARROW (the main feature!)
                        arrow_start_frame = source_last + 1
                        arrow_end_frame = target_first - 1

                        if arrow_end_frame > arrow_start_frame:
                            # Draw curved arrow
                            self.ax5.annotate('',
                                            xy=(arrow_end_frame, target_y),
                                            xytext=(arrow_start_frame, source_y),
                                            arrowprops=dict(arrowstyle='->',
                                                          connectionstyle='arc3,rad=0.3',
                                                          color='purple', lw=3, alpha=0.9))

                            # Add reconciliation label
                            mid_frame = (arrow_start_frame + arrow_end_frame) / 2
                            mid_y = (source_y + target_y) / 2
                            self.ax5.text(mid_frame, mid_y, f'{source_id}→{target_id}',
                                         ha='center', va='center',
                                         bbox=dict(boxstyle='round,pad=0.3',
                                                  facecolor='purple', alpha=0.8),
                                         color='white', fontweight='bold', fontsize=8)

                        # Update y_positions to include source IDs
                        y_positions[source_id] = source_y

        # Add ephemeral IDs that were filtered out (shown in red at top)
        ephemeral_y_start = len(sorted_ids) + len(self.reconciliation_mapping) + 1
        for i, ephemeral_id in enumerate(sorted(self.ephemeral_ids)):
            if ephemeral_id not in self.reconciliation_mapping:  # Don't show if already reconciled
                frames = self.all_id_history.get(ephemeral_id, [])
                if frames:
                    y_pos = ephemeral_y_start + i
                    self.ax5.plot([min(frames), max(frames)], [y_pos, y_pos],
                                 color='red', linewidth=1, alpha=0.5, linestyle=':')
                    self.ax5.text(min(frames) - 5, y_pos, f'{ephemeral_id}',
                                 va='center', ha='right', fontweight='bold',
                                 color='red', fontsize=8)

        # Add legend and annotations
        legend_elements = [
            plt.Line2D([0], [0], color='purple', lw=4, label='Reconciled Target ID'),
            plt.Line2D([0], [0], color='green', lw=3, label='Persistent ID'),
            plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Reconciled Source ID'),
            plt.Line2D([0], [0], color='red', lw=1, linestyle=':', label='Ephemeral ID (Filtered)')
        ]
        self.ax5.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Add text annotation
        if self.reconciliation_mapping:
            self.ax5.text(0.02, 0.98, f'🔄 {len(self.reconciliation_mapping)} reconciliation(s) performed',
                         transform=self.ax5.transAxes, va='top', ha='left',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                         fontweight='bold')

    def _update_reconciliation_statistics(self):
        """Update reconciliation statistics visualization."""
        self.ax6.clear()
        self.ax6.set_title('📊 Reconciliation Statistics & Impact')

        # Prepare data
        stages = ['Raw\nDetection', 'Ephemeral\nFilter', 'Local\nReconciliation']

        # Calculate counts
        total_ids = len(self.all_unique_ids)
        persistent_ids = len(self.persistent_ids)
        reconciled_pairs = len(self.reconciliation_mapping)
        final_ids = persistent_ids  # After reconciliation, source IDs are merged

        counts = [total_ids, persistent_ids + reconciled_pairs, final_ids]
        colors = ['lightblue', 'lightgreen', 'purple']

        # Create bar chart
        bars = self.ax6.bar(stages, counts, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            self.ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         str(count), ha='center', va='bottom', fontweight='bold')

        # Add reduction arrows and percentages
        if len(counts) > 1:
            for i in range(len(counts) - 1):
                reduction = counts[i] - counts[i+1]
                if reduction > 0:
                    reduction_pct = (reduction / counts[i]) * 100
                    mid_x = i + 0.5
                    mid_y = max(counts) * 0.7

                    self.ax6.annotate(f'-{reduction}\n({reduction_pct:.1f}%)',
                                    xy=(mid_x, mid_y), ha='center', va='center',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.8),
                                    fontweight='bold', fontsize=9)

        # Add summary text
        efficiency_text = ""
        if total_ids > 0:
            filter_efficiency = (len(self.ephemeral_ids) / total_ids) * 100
            reconciliation_efficiency = (reconciled_pairs / max(1, persistent_ids + reconciled_pairs)) * 100

            efficiency_text = f"Filter Efficiency: {filter_efficiency:.1f}%\n"
            efficiency_text += f"Reconciliation Rate: {reconciliation_efficiency:.1f}%"

        self.ax6.text(0.02, 0.98, efficiency_text,
                     transform=self.ax6.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                     fontsize=10)

        self.ax6.set_ylabel('Number of IDs')
        self.ax6.grid(True, alpha=0.3)

    def process_video(self):
        """Process video with YOLO tracking and ephemeral filtering."""
        print("Starting YOLO tracking with ephemeral filtering...")
        print(f"Ephemeral threshold: {self.ephemeral_threshold} frames")
        print("Press 'q' in the video window to quit")

        # Phase 1: Collect all tracking data
        print("\n=== Phase 1: Collecting tracking data ===")
        results = self.model.track(
            self.video_path,
            show=False,  # We'll handle video display manually
            tracker="bytetrack.yaml",
            classes=[0],  # Only detect persons
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            stream=True,
            verbose=False,
            conf=0.35,
            iou=0.5
        )

        # Open video for manual display with annotations
        cap = cv2.VideoCapture(self.video_path)

        for result in results:
            self.current_frame += 1

            # Get the current video frame
            ret, frame = cap.read()
            if not ret:
                break

            # Count people in current frame
            person_count = 0
            frame_detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                person_count = len(result.boxes)

                # Extract tracking IDs and bounding boxes
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    xyxy = result.boxes.xyxy.cpu().numpy()  # Bounding boxes

                    # Store detections for this frame
                    for track_id, bbox in zip(track_ids, xyxy):
                        x1, y1, x2, y2 = bbox.astype(int)
                        frame_detections.append({
                            'track_id': track_id,
                            'bbox': (x1, y1, x2, y2)
                        })

                        # Update tracking history and frame counts
                        self.all_unique_ids.add(track_id)
                        self.all_id_history[track_id].append(self.current_frame)
                        self.id_frame_counts[track_id] += 1

                        # Store bounding box data for local reconciliation
                        self.id_bounding_boxes[track_id].append((x1, y1, x2, y2))

                        # Update first and last frame for each ID
                        if track_id not in self.id_first_frame:
                            self.id_first_frame[track_id] = self.current_frame
                        self.id_last_frame[track_id] = self.current_frame

            # Store frame detections for video annotation
            self.frame_detections[self.current_frame] = frame_detections

            # Store frame data
            self.frame_numbers.append(self.current_frame)
            self.num_of_people_in_each_frame.append(person_count)

            # Re-evaluate ephemeral status periodically
            if self.current_frame % self.reevaluation_interval == 0:
                self._identify_ephemeral_ids()

            # Draw bounding boxes on frame
            frame_copy = frame.copy()
            for detection in frame_detections:
                track_id = detection['track_id']
                x1, y1, x2, y2 = detection['bbox']

                # Get color and label based on ephemeral status
                color, label = self._get_box_color_and_label(track_id)

                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                self._draw_text_with_background(frame_copy, label, (x1, y1), bg_color=color)

            # Add frame info
            info_text = f"Frame: {self.current_frame} | People: {person_count} | "
            info_text += f"All IDs: {len(self.all_unique_ids)} | Persistent: {len(self.persistent_ids)} | "
            info_text += f"Ephemeral: {len(self.ephemeral_ids)}"
            self._draw_text_with_background(frame_copy, info_text, (10, 30), font_scale=0.7, bg_color=(0, 0, 0))

            # Show annotated frame
            cv2.imshow('YOLO Tracking with Ephemeral Filtering', frame_copy)

            # Update plots every 10 frames for performance
            if self.current_frame % 10 == 0:
                self.update_plots()

            # Print current stats
            if self.current_frame % 50 == 0:
                print(f"Frame {self.current_frame}: {person_count} people, "
                      f"Total IDs: {len(self.all_unique_ids)}, "
                      f"Persistent: {len(self.persistent_ids)}, "
                      f"Ephemeral: {len(self.ephemeral_ids)}")

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Phase 2: Final ephemeral filtering and local reconciliation
        print("\n=== Phase 2: Final ephemeral filtering and local reconciliation ===")
        self._identify_ephemeral_ids()

        # Final plot update
        self.update_plots()

        # Print summary statistics
        self.print_summary()

        # Keep plots open
        print("\nVisualization complete! Close the plot window to exit.")
        plt.ioff()
        plt.show()

    def print_summary(self):
        """Print detailed summary statistics."""
        print("\n" + "="*70)
        print("TRACKING SUMMARY WITH EPHEMERAL FILTERING")
        print("="*70)
        print(f"Total frames processed: {self.current_frame}")
        print(f"Ephemeral threshold: {self.ephemeral_threshold} frames")
        print()

        print("ALL TRACKING RESULTS:")
        print(f"  • Average people per frame: {np.mean(self.num_of_people_in_each_frame):.2f}")
        print(f"  • Max people in single frame: {max(self.num_of_people_in_each_frame) if self.num_of_people_in_each_frame else 0}")
        print(f"  • Min people in single frame: {min(self.num_of_people_in_each_frame) if self.num_of_people_in_each_frame else 0}")
        print(f"  • Total unique IDs tracked: {len(self.all_unique_ids)}")
        print()

        print("EPHEMERAL FILTERING RESULTS:")
        print(f"  • Persistent IDs (≥{self.ephemeral_threshold} frames): {len(self.persistent_ids)}")
        print(f"  • Ephemeral IDs (<{self.ephemeral_threshold} frames): {len(self.ephemeral_ids)}")
        print(f"  • Filtering efficiency: {len(self.ephemeral_ids)/len(self.all_unique_ids)*100:.1f}% IDs removed")
        print()

        print("LOCAL ID RECONCILIATION RESULTS:")
        reconciled_count = len(self.reconciliation_mapping)
        print(f"  • ID pairs reconciled: {reconciled_count}")
        if reconciled_count > 0:
            print(f"  • Reconciliation pairs:")
            for old_id, new_id in sorted(self.reconciliation_mapping.items()):
                print(f"    - ID {old_id} merged into ID {new_id}")
        print(f"  • Final persistent IDs after reconciliation: {len(self.persistent_ids)}")
        print()

        if self.filtered_num_of_people_in_each_frame:
            print(f"FILTERED TRACKING STATS:")
            print(f"  • Average persistent people per frame: {np.mean(self.filtered_num_of_people_in_each_frame):.2f}")
            print()

        if self.persistent_id_history:
            print("PERSISTENT ID ANALYSIS:")
            for track_id, frames in sorted(self.persistent_id_history.items()):
                lifespan = len(frames)
                first_frame = min(frames)
                last_frame = max(frames)
                print(f"  ID {track_id}: appeared in {lifespan} frames (frames {first_frame}-{last_frame})")

        if self.ephemeral_ids:
            print(f"\nEPHEMERAL IDs REMOVED: {sorted(list(self.ephemeral_ids))}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='YOLO Tracking Visualization with Ephemeral Filtering')
    parser.add_argument('--ephemeral_threshold', type=int, default=30,
                       help='Minimum frames an ID must appear to be considered persistent (default: 30)')
    parser.add_argument('--model_path', type=str, default="../yolo11l.pt",
                       help='Path to YOLO model weights (default: ../yolo11n.pt)')
    parser.add_argument('--video_path', type=str, default="../videos/8th-grade-vid.mp4",
                       help='Path to input video (default: ../videos/8th-grade-vid.mp4')
    parser.add_argument('--reevaluation_interval', type=int, default=10,
                       help='How often to re-evaluate ephemeral status in frames (default: 30)')
    parser.add_argument('--temporal_gap_threshold', type=int, default=35,
                       help='Maximum frame gap for temporal matching in local reconciliation (default: 10)')
    parser.add_argument('--min_iou_threshold', type=float, default=0.3,
                       help='Minimum IoU threshold for spatial matching in local reconciliation (default: 0.3)')

    args = parser.parse_args()

    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using ephemeral threshold: {args.ephemeral_threshold} frames")
    print(f"Using re-evaluation interval: {args.reevaluation_interval} frames")
    print(f"Using temporal gap threshold: {args.temporal_gap_threshold} frames")
    print(f"Using minimum IoU threshold: {args.min_iou_threshold}")

    visualizer = YOLOTrackingVisualizerWithEphemeralFiltering(
        model_path=args.model_path,
        video_path=args.video_path,
        ephemeral_threshold=args.ephemeral_threshold,
        reevaluation_interval=args.reevaluation_interval,
        temporal_gap_threshold=args.temporal_gap_threshold,
        min_iou_threshold=args.min_iou_threshold
    )
    visualizer.process_video()


if __name__ == "__main__":
    main()
