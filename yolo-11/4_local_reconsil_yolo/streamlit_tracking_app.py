#!/usr/bin/env python3
"""
Streamlit Web Portal for YOLO Tracking with Ephemeral Filtering and Local ID Reconciliation
===========================================================================================
Simplified web interface for remote access.

Usage:
    streamlit run streamlit_tracking_app.py
"""

import streamlit as st
import cv2
import numpy as np
import torch
from collections import defaultdict
from ultralytics import YOLO
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TrackingEngine:
    """Core tracking logic without UI dependencies."""
    
    def __init__(self, model_path, video_path, ephemeral_threshold=30, 
                 reevaluation_interval=30, temporal_gap_threshold=10, min_iou_threshold=0.3):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.ephemeral_threshold = ephemeral_threshold
        self.reevaluation_interval = reevaluation_interval
        self.temporal_gap_threshold = temporal_gap_threshold
        self.min_iou_threshold = min_iou_threshold
        
        # Tracking data
        self.num_of_people_in_each_frame = []
        self.frame_numbers = []
        self.all_unique_ids = set()
        self.all_id_history = defaultdict(list)
        self.current_frame = 0
        
        # Filtered data
        self.filtered_num_of_people_in_each_frame = []
        self.persistent_ids = set()
        self.persistent_id_history = defaultdict(list)
        
        # Ephemeral filtering
        self.ephemeral_ids = set()
        self.id_frame_counts = defaultdict(int)
        self.frame_detections = {}
        
        # Local reconciliation
        self.id_bounding_boxes = defaultdict(list)
        self.id_first_frame = {}
        self.id_last_frame = {}
        self.reconciliation_mapping = {}
        self.reconciled_num_of_people_in_each_frame = []

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

    def _get_representative_bbox(self, track_id):
        if track_id not in self.id_bounding_boxes or not self.id_bounding_boxes[track_id]:
            return None
        boxes = self.id_bounding_boxes[track_id]
        x1_median = sorted([box[0] for box in boxes])[len(boxes) // 2]
        y1_median = sorted([box[1] for box in boxes])[len(boxes) // 2]
        x2_median = sorted([box[2] for box in boxes])[len(boxes) // 2]
        y2_median = sorted([box[3] for box in boxes])[len(boxes) // 2]
        return (x1_median, y1_median, x2_median, y2_median)

    def _perform_local_reconciliation(self):
        self.reconciliation_mapping = {}
        persistent_ids_list = sorted(list(self.persistent_ids),
                                    key=lambda x: self.id_first_frame.get(x, float('inf')))
        
        for i, id1 in enumerate(persistent_ids_list):
            if id1 in self.reconciliation_mapping:
                continue
            
            id1_last = self.id_last_frame.get(id1, 0)
            id1_bbox = self._get_representative_bbox(id1)
            if id1_bbox is None:
                continue
            
            best_match = None
            best_iou = 0.0
            
            for id2 in persistent_ids_list[i+1:]:
                if id2 in self.reconciliation_mapping:
                    continue
                
                id2_first = self.id_first_frame.get(id2, float('inf'))
                temporal_gap = id2_first - id1_last
                
                if temporal_gap <= 0 or temporal_gap > self.temporal_gap_threshold:
                    continue
                
                id2_bbox = self._get_representative_bbox(id2)
                if id2_bbox is None:
                    continue
                
                iou = self._calculate_iou(id1_bbox, id2_bbox)
                if iou >= self.min_iou_threshold and iou > best_iou:
                    best_match = id2
                    best_iou = iou
            
            if best_match is not None:
                self.reconciliation_mapping[best_match] = id1
        
        if self.reconciliation_mapping:
            self._apply_reconciliation_mapping()
            self._build_reconciled_tracking_data()

    def _apply_reconciliation_mapping(self):
        if not self.reconciliation_mapping:
            return
        
        new_all_id_history = defaultdict(list)
        for track_id, frames in self.all_id_history.items():
            target_id = self.reconciliation_mapping.get(track_id, track_id)
            new_all_id_history[target_id].extend(frames)
        
        for track_id in new_all_id_history:
            new_all_id_history[track_id] = sorted(list(set(new_all_id_history[track_id])))
        
        self.all_id_history = new_all_id_history
        
        new_id_frame_counts = defaultdict(int)
        for track_id, frames in self.all_id_history.items():
            new_id_frame_counts[track_id] = len(frames)
        self.id_frame_counts = new_id_frame_counts
        
        ids_to_remove = set(self.reconciliation_mapping.keys())
        self.persistent_ids = self.persistent_ids - ids_to_remove
        self.all_unique_ids = self.all_unique_ids - ids_to_remove
        
        for track_id, frames in self.all_id_history.items():
            if frames:
                self.id_first_frame[track_id] = min(frames)
                self.id_last_frame[track_id] = max(frames)
        
        for reconciled_id in ids_to_remove:
            self.id_first_frame.pop(reconciled_id, None)
            self.id_last_frame.pop(reconciled_id, None)
            self.id_bounding_boxes.pop(reconciled_id, None)
        
        self._build_persistent_tracking_data()

    def _build_reconciled_tracking_data(self):
        self.reconciled_num_of_people_in_each_frame = []
        for frame_num in self.frame_numbers:
            reconciled_count = sum(1 for track_id, frames in self.persistent_id_history.items()
                                 if frame_num in frames)
            self.reconciled_num_of_people_in_each_frame.append(reconciled_count)

    def _identify_ephemeral_ids(self):
        self.ephemeral_ids = {track_id for track_id, count in self.id_frame_counts.items()
                             if count < self.ephemeral_threshold}
        self.persistent_ids = self.all_unique_ids - self.ephemeral_ids
        self._build_persistent_tracking_data()
        
        if len(self.persistent_ids) > 1:
            self._perform_local_reconciliation()

    def _build_persistent_tracking_data(self):
        self.persistent_id_history = {track_id: frames for track_id, frames in self.all_id_history.items()
                                     if track_id in self.persistent_ids}
        
        self.filtered_num_of_people_in_each_frame = []
        for frame_num in self.frame_numbers:
            persistent_count = sum(1 for track_id, frames in self.persistent_id_history.items()
                                 if frame_num in frames)
            self.filtered_num_of_people_in_each_frame.append(persistent_count)

    def _get_box_color_and_label(self, track_id):
        if track_id in self.reconciliation_mapping:
            target_id = self.reconciliation_mapping[track_id]
            return (255, 0, 255), f"ID: {track_id}→{target_id}"
        elif track_id in self.ephemeral_ids:
            return (0, 0, 255), f"ID: {track_id} (Removed)"
        elif track_id in self.persistent_ids:
            return (0, 255, 0), f"ID: {track_id}"
        else:
            return (0, 255, 255), f"ID: {track_id} (Eval)"

    def process_frame(self, result, frame):
        """Process a single frame and return annotated frame."""
        self.current_frame += 1
        person_count = 0
        frame_detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            person_count = len(result.boxes)
            
            if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                xyxy = result.boxes.xyxy.cpu().numpy()
                
                for track_id, bbox in zip(track_ids, xyxy):
                    x1, y1, x2, y2 = bbox.astype(int)
                    frame_detections.append({'track_id': track_id, 'bbox': (x1, y1, x2, y2)})
                    
                    self.all_unique_ids.add(track_id)
                    self.all_id_history[track_id].append(self.current_frame)
                    self.id_frame_counts[track_id] += 1
                    self.id_bounding_boxes[track_id].append((x1, y1, x2, y2))
                    
                    if track_id not in self.id_first_frame:
                        self.id_first_frame[track_id] = self.current_frame
                    self.id_last_frame[track_id] = self.current_frame
        
        self.frame_detections[self.current_frame] = frame_detections
        self.frame_numbers.append(self.current_frame)
        self.num_of_people_in_each_frame.append(person_count)
        
        if self.current_frame % self.reevaluation_interval == 0:
            self._identify_ephemeral_ids()
        
        # Draw annotations
        for detection in frame_detections:
            track_id = detection['track_id']
            x1, y1, x2, y2 = detection['bbox']
            color, label = self._get_box_color_and_label(track_id)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label) * 10, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        info_text = f"Frame: {self.current_frame} | People: {person_count} | IDs: {len(self.all_unique_ids)} | Persistent: {len(self.persistent_ids)} | Ephemeral: {len(self.ephemeral_ids)}"
        cv2.rectangle(frame, (5, 5), (len(info_text) * 8, 35), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


def create_plotly_figures(engine):
    """Create all 6 Plotly figures."""
    
    # Figure 1: People count comparison
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=engine.frame_numbers, y=engine.num_of_people_in_each_frame,
                             mode='lines', name='Raw', line=dict(color='blue', width=2)))
    if engine.filtered_num_of_people_in_each_frame:
        fig1.add_trace(go.Scatter(x=engine.frame_numbers, y=engine.filtered_num_of_people_in_each_frame,
                                 mode='lines', name='Filtered', line=dict(color='green', width=2)))
    if engine.reconciled_num_of_people_in_each_frame:
        fig1.add_trace(go.Scatter(x=engine.frame_numbers, y=engine.reconciled_num_of_people_in_each_frame,
                                 mode='lines', name='Reconciled', line=dict(color='purple', width=3)))
    fig1.update_layout(title='People Count Pipeline', xaxis_title='Frame', yaxis_title='Count', height=300)
    
    # Figure 2: ID tracking scatter
    fig2 = go.Figure()
    for track_id, frames in engine.all_id_history.items():
        if frames:
            if track_id in engine.reconciliation_mapping:
                color, marker = 'purple', 'square'
            elif track_id in engine.ephemeral_ids:
                color, marker = 'red', 'x'
            elif track_id in engine.persistent_ids:
                color, marker = 'green', 'circle'
            else:
                color, marker = 'orange', 'triangle-up'
            fig2.add_trace(go.Scatter(x=frames, y=[track_id]*len(frames), mode='markers',
                                     marker=dict(color=color, symbol=marker, size=8),
                                     name=f'ID {track_id}', showlegend=False))
    fig2.update_layout(title='ID Tracking Over Time', xaxis_title='Frame', yaxis_title='ID', height=300)
    
    # Figure 3: Processing pipeline
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=engine.frame_numbers, y=engine.num_of_people_in_each_frame,
                             mode='lines', name='Raw', line=dict(color='blue', width=1, dash='dot')))
    if engine.filtered_num_of_people_in_each_frame:
        fig3.add_trace(go.Scatter(x=engine.frame_numbers, y=engine.filtered_num_of_people_in_each_frame,
                                 mode='lines', name='Filtered', line=dict(color='green', width=2)))
    if engine.reconciled_num_of_people_in_each_frame:
        fig3.add_trace(go.Scatter(x=engine.frame_numbers, y=engine.reconciled_num_of_people_in_each_frame,
                                 mode='lines', name='Final', line=dict(color='purple', width=3)))
    fig3.update_layout(title='Processing Pipeline', xaxis_title='Frame', yaxis_title='Count', height=300)
    
    # Figure 4: Reconciliation mappings
    fig4 = go.Figure()
    if engine.reconciliation_mapping:
        for source_id, target_id in engine.reconciliation_mapping.items():
            if source_id in engine.all_id_history and target_id in engine.all_id_history:
                source_frames = engine.all_id_history[source_id]
                target_frames = engine.all_id_history[target_id]
                fig4.add_trace(go.Scatter(x=source_frames, y=[source_id]*len(source_frames),
                                         mode='markers', marker=dict(color='red', symbol='x'),
                                         name=f'{source_id}', showlegend=False))
                combined = sorted(set(source_frames + target_frames))
                fig4.add_trace(go.Scatter(x=combined, y=[target_id]*len(combined),
                                         mode='markers', marker=dict(color='purple', size=10),
                                         name=f'{target_id}', showlegend=False))
    fig4.update_layout(title=f'Reconciliation: {len(engine.reconciliation_mapping)} pairs', 
                      xaxis_title='Frame', yaxis_title='ID', height=300)
    
    # Figure 5: Temporal view
    fig5 = go.Figure()
    sorted_ids = sorted(list(engine.persistent_ids), key=lambda x: engine.id_first_frame.get(x, float('inf')))
    for i, track_id in enumerate(sorted_ids):
        if track_id in engine.persistent_id_history:
            frames = engine.persistent_id_history[track_id]
            color = 'purple' if track_id in engine.reconciliation_mapping.values() else 'green'
            fig5.add_trace(go.Scatter(x=[min(frames), max(frames)], y=[i, i],
                                     mode='lines', line=dict(color=color, width=4),
                                     name=f'ID {track_id}', showlegend=False))
    fig5.update_layout(title='Temporal View', xaxis_title='Frame', yaxis_title='ID Index', height=300)
    
    # Figure 6: Statistics
    fig6 = go.Figure()
    total_ids = len(engine.all_unique_ids)
    persistent_ids = len(engine.persistent_ids)
    reconciled_pairs = len(engine.reconciliation_mapping)
    final_ids = persistent_ids
    
    stages = ['Raw', 'Filtered', 'Reconciled']
    counts = [total_ids, persistent_ids + reconciled_pairs, final_ids]
    colors = ['lightblue', 'lightgreen', 'purple']
    
    fig6.add_trace(go.Bar(x=stages, y=counts, marker_color=colors, text=counts, textposition='outside'))
    fig6.update_layout(title='Statistics', xaxis_title='Stage', yaxis_title='ID Count', height=300)
    
    return fig1, fig2, fig3, fig4, fig5, fig6


# Streamlit UI
st.set_page_config(page_title="YOLO Tracking Portal", layout="wide", initial_sidebar_state="expanded")
st.title("📹 YOLO Tracking - Ephemeral Filtering & Local ID Reconciliation")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    model_path = st.text_input("Model Path", "../yolo11l.pt")
    video_path = st.text_input("Video Path", "../videos/cmu_05391A_407sc_104_20241003152848_front.avi")
    
    st.subheader("Parameters")
    ephemeral_threshold = st.slider("Ephemeral Threshold", 10, 100, 30)
    reevaluation_interval = st.slider("Reevaluation Interval", 5, 50, 10)
    temporal_gap_threshold = st.slider("Temporal Gap", 5, 50, 35)
    min_iou_threshold = st.slider("Min IoU", 0.1, 0.9, 0.3)
    
    start_button = st.button("▶️ Start Processing", type="primary")
    stop_button = st.button("⏹️ Stop")
    
    st.divider()
    st.subheader("📊 Live Stats")
    stats_placeholder = st.empty()

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'engine' not in st.session_state:
    st.session_state.engine = None

# Control logic
if start_button:
    st.session_state.processing = True
    st.session_state.engine = TrackingEngine(
        model_path, video_path, ephemeral_threshold,
        reevaluation_interval, temporal_gap_threshold, min_iou_threshold
    )

if stop_button:
    st.session_state.processing = False

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Video Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📈 Quick Stats")
    metrics_placeholder = st.empty()

# Plots
st.divider()
plot_cols = st.columns(2)
plot1, plot2, plot3, plot4, plot5, plot6 = [col.empty() for col in plot_cols for _ in range(3)]

# Processing loop
if st.session_state.processing and st.session_state.engine:
    engine = st.session_state.engine
    
    results = engine.model.track(
        engine.video_path,
        show=False,
        tracker="bytetrack.yaml",
        classes=[0],
        device='mps' if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu',
        stream=True,
        verbose=False,
        conf=0.35,
        iou=0.5
    )
    
    cap = cv2.VideoCapture(engine.video_path)
    progress_bar = st.progress(0)
    
    for result in results:
        if not st.session_state.processing:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame = engine.process_frame(result, frame)
        
        # Update video
        video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Update metrics
        with metrics_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Frame", engine.current_frame)
            m2.metric("Total IDs", len(engine.all_unique_ids))
            m3.metric("Persistent", len(engine.persistent_ids))
            m4.metric("Ephemeral", len(engine.ephemeral_ids))
        
        # Update plots every 10 frames
        if engine.current_frame % 10 == 0:
            fig1, fig2, fig3, fig4, fig5, fig6 = create_plotly_figures(engine)
            with plot_cols[0]:
                plot1.plotly_chart(fig1, use_container_width=True, key=f"plot1_{engine.current_frame}")
                plot3.plotly_chart(fig3, use_container_width=True, key=f"plot3_{engine.current_frame}")
                plot5.plotly_chart(fig5, use_container_width=True, key=f"plot5_{engine.current_frame}")
            with plot_cols[1]:
                plot2.plotly_chart(fig2, use_container_width=True, key=f"plot2_{engine.current_frame}")
                plot4.plotly_chart(fig4, use_container_width=True, key=f"plot4_{engine.current_frame}")
                plot6.plotly_chart(fig6, use_container_width=True, key=f"plot6_{engine.current_frame}")
        
        # Update sidebar stats
        with stats_placeholder.container():
            st.metric("Reconciled Pairs", len(engine.reconciliation_mapping))
            if len(engine.all_unique_ids) > 0:
                filter_eff = (len(engine.ephemeral_ids) / len(engine.all_unique_ids)) * 100
                st.metric("Filter Efficiency", f"{filter_eff:.1f}%")
    
    cap.release()
    
    # Final update
    engine._identify_ephemeral_ids()
    fig1, fig2, fig3, fig4, fig5, fig6 = create_plotly_figures(engine)
    with plot_cols[0]:
        plot1.plotly_chart(fig1, use_container_width=True, key="plot1_final")
        plot3.plotly_chart(fig3, use_container_width=True, key="plot3_final")
        plot5.plotly_chart(fig5, use_container_width=True, key="plot5_final")
    with plot_cols[1]:
        plot2.plotly_chart(fig2, use_container_width=True, key="plot2_final")
        plot4.plotly_chart(fig4, use_container_width=True, key="plot4_final")
        plot6.plotly_chart(fig6, use_container_width=True, key="plot6_final")
    
    progress_bar.progress(100)
    st.session_state.processing = False
    st.success("✅ Processing Complete!")

