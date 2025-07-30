# # Original command line usage
# python script.py Fall2023 CS101 0
# #                  ↑        ↑   ↑
# #              semester  course gpu
# script.py = tracking_pipeline_singlethread.py


# Multi-object tracking (the heart of student ID assignment)
import os
import sys
from utils_computev3 import get_logger, time_diff
from mmtrack.apis import inference_mot, init_model as init_tracking_model 
import mmcv
from datetime import datetime
from utils_computev3 import time_diff, get_logger
import time
import torch
import time
import pickle
import glob
import tensorflow as tf
import shutil

# Imports that are not used anywhere in the script: ------------------------------------------------------
from configs.get_session_config import get_session_config
from FaceWrapper import RetinaFaceInference
from concurrent.futures import ThreadPoolExecutor
from GazeWrapper import GazeInference
from facenet_pytorch import InceptionResnetV1
from queue import Empty as EmptyQueueException
from subprocess import Popen
import threading
import argparse
import cv2
import numpy as np
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
# --------------------------------------------------------------------------------------------------------



tf_version = tf.__version__
tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image






# sys.argv[1]: Institution/semester identifier
# sys.argv[2]: Course ID (e.g., "CS101")
# sys.argv[3]: GPU device number


# cd /home/prasoon/video_analysis/edusenseV2compute/compute/videoV3/ && conda activate edusense && export PYTHONPATH="${PYTHONPATH}:/home/prasoon/video_analysis/edusenseV2compute/compute/videoV3/"
SOURCE_DIR = '/home/prasoon/video_analysis/edusenseV2compute/compute/videoV3'
VIDEO_DIR = f'/mnt/ci-nas-classes/classinsight/{sys.argv[1]}/video_backup'
COURSE_ID = sys.argv[2]
DEVICE_ARG = int(sys.argv[3])
DEVICE = f'cuda:{DEVICE_ARG}'


TARGET_FPS = 5                  # Process 5 frames per second (downsampling for efficiency)
# FRAME_INTERVAL_IN_SEC = 0.5  Not used anywhere!


session_dirs = glob.glob(f'{VIDEO_DIR}/*')
print(f"Got {len(session_dirs)} sessions from {VIDEO_DIR}.")
session_dirs = [xr for xr in session_dirs if f'_{COURSE_ID}_' in xr]
print(f"Got {len(session_dirs)} sessions from {VIDEO_DIR} and course {COURSE_ID}")

# Find all directories in the video backup folder
# Filter to only sessions matching the specific course ID
# Each session directory contains video files from that class period

# Example structure of VIDEO_DIR:
# /mnt/ci-nas-classes/classinsight/Fall2023/video_backup/
# ├── 2023-09-15_CS101_Session1/
# ├── 2023-09-22_CS101_Session2/
# ├── 2023-09-15_MATH201_Session1/  # Filtered out
# └── 2023-09-29_CS101_Session3/





if __name__ == '__main__':


    for SESSION_DIR in session_dirs: # Outer Loop: Each class session (e.g., different dates)
        SESSION_KEYWORD = SESSION_DIR.split("/")[-1]
        
        for SESSION_CAMERA in ['front']: # Each camera angle (currently only 'front')

            SESSION_CAMERA_FILES = glob.glob(f'{SESSION_DIR}/*{SESSION_CAMERA}.avi')
            
            for SESSION_CAMERA_FILE in SESSION_CAMERA_FILES: # Each video file for that camera
                
                
                SESSION_KEYWORD = \
                SESSION_CAMERA_FILE.split("/")[-1].split(f"_{SESSION_CAMERA}.avi")[0].split(f"-{SESSION_CAMERA}.avi")[0]
                
                # get logger -----------------------------------------------------------------------
                session_log_dir = f'cache/tracking_singlethread_only/logs/{COURSE_ID}'
                os.makedirs(session_log_dir, exist_ok=True)
                logger = get_logger(f"{SESSION_KEYWORD}-{SESSION_CAMERA}", logdir=session_log_dir)
                logger.info(f"processing for session {SESSION_KEYWORD}, {SESSION_CAMERA}")
                # ----------------------------------------------------------------------------------

                # initialize session config
                # smart resume logic:
                    # If end.pickle exists: Session already completed, skip it
                    # If directory exists but no end.pickle: Previous run failed, restart
                    # Creates clean output directory for each session
                session_frames_output_dir = f'cache/tracking_only/output/{COURSE_ID}/{SESSION_KEYWORD}-{SESSION_CAMERA}'
                if os.path.exists(session_frames_output_dir):
                    if os.path.exists(f"{session_frames_output_dir}/end.pickle"):
                        logger.info(f"session tracking output dir exists with end file, Delete it to rerun the session.")
                        continue
                    else:
                        logger.info(
                            f"session tracking output dir exists but no end file, Deleting it and rerunning the session.")
                        shutil.rmtree(session_frames_output_dir)
                os.makedirs(session_frames_output_dir, exist_ok=True)
                t_start_session = time.time()
                # setup networks for tracking handlers

                # -------Tracking Model init-------
                # YOLOX: Object detection backbone (finds people in images)
                # OC-SORT: Observation-Centric SORT tracking algorithm
                # CrowdHuman: Training dataset for crowded scenarios
                # MOT17: Multi-Object Tracking benchmark dataset
                
                # Check here: https://github.com/open-mmlab/mmtracking/tree/master/configs
                # Check here for "ocsort_yolox_x_crowdhuman_mot17-private-half.py": https://github.com/open-mmlab/mmdetection/pull/11533/files
                
                tracking_config = {
                    # trackingHandler
                    'track_config': f'{SOURCE_DIR}/configs/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half.py',
                    'track_checkpoint': f'{SOURCE_DIR}/models/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth',
                }

                tracking_model = init_tracking_model(tracking_config['track_config'],
                                                     tracking_config['track_checkpoint'],
                                                     device=DEVICE)

                # start loop with frames and video handler

                # class_video_file = f'{SESSION_DIR}/{SESSION_KEYWORD}-{SESSION_CAMERA}.avi'    redundant right?
                class_video_file = SESSION_CAMERA_FILE
                if not os.path.exists(class_video_file):
                    logger.info(f"Video File {class_video_file} not available, skipping session...")
                    continue
                mmcv_video_frames = mmcv.VideoReader(class_video_file)
                
                # Original video might be 30 FPS
                # Target processing: 5 FPS
                # num_skip_frames = 6 means process every 6th frame
                video_fps = mmcv_video_frames.fps
                # h, w, _ = mmcv_video_frames[0].shape
                logger.info("reading frames from video")
                num_skip_frames = int(video_fps / TARGET_FPS)



                session_process_start = datetime.now()
                for frame_number, video_frame in enumerate(mmcv_video_frames):
                    if (num_skip_frames == 0) | (frame_number % num_skip_frames == 0):
                        
                        # get tracking output
                        track_process_start = datetime.now()

                        track_results = inference_mot(tracking_model, video_frame, frame_id=frame_number)
                        # List of detected people with bbox and track_id
                        track_bboxes = track_results['track_bboxes'][0]
                        track_results = [dict(bbox=x[1:], track_id=x[0]) for x in list(track_bboxes)]
                        
                        track_process_end = datetime.now()
                        
                        
                        logger.info(
                            f"Frame: {frame_number} | track | {time_diff(track_process_start, track_process_end):.3f} secs")
                        # output frame in tracking only dir
                        pickle.dump((frame_number, track_results),
                                    open(f'{session_frames_output_dir}/{frame_number}.pickle', 'wb'))
                
                
                
                pickle.dump((frame_number, track_results), open(f'{session_frames_output_dir}/end.pickle', 'wb'))
                del tracking_model, mmcv_video_frames, logger
                time.sleep(5)
                torch.cuda.empty_cache()
