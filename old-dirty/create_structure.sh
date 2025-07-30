#!/bin/bash

# Set root directory name
ROOT_DIR="classid_project"

# Create directory structure
mkdir -p $ROOT_DIR/data
mkdir -p $ROOT_DIR/output/session_1/raw_features
mkdir -p $ROOT_DIR/output/session_2
mkdir -p $ROOT_DIR/pipeline

# Create placeholder files
touch $ROOT_DIR/main_orchestrator.py

# Sample data files (optional placeholder video files)
touch $ROOT_DIR/data/session_1.mp4
touch $ROOT_DIR/data/session_2.mp4

# Output files
touch $ROOT_DIR/output/session_1/id_map.json
touch $ROOT_DIR/output/session_1/session_reps.json

# Pipeline files
touch $ROOT_DIR/pipeline/__init__.py
touch $ROOT_DIR/pipeline/step_1_extract_features.py
touch $ROOT_DIR/pipeline/step_2_reconcile_ids.py
touch $ROOT_DIR/pipeline/step_3_create_representation.py
touch $ROOT_DIR/pipeline/utils.py

echo "Directory structure created successfully."