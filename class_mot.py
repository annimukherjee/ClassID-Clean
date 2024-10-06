
import os
import sys

from configs.get_session_config import get_session_config
from utils_computev3 import get_logger, time_diff
from mmtrack.apis import inference_mot, init_model as init_tracking_model
import mmcv
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from datetime import datetime
from FaceWrapper import RetinaFaceInference
from utils import time_diff, get_logger
from concurrent.futures import ThreadPoolExecutor
from GazeWrapper import GazeInference
from facenet_pytorch import InceptionResnetV1
import time
import os
import argparse
import torch
import threading
from subprocess import Popen
import numpy as np
from queue import Empty as EmptyQueueException
import time
import pickle
import glob
import tensorflow as tf
import cv2
import shutil



video_filepath = sys.argv[1]

TARGET_FPS = 5
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
video_id = video_filepath.split('/')[-1].split('.')[0]
# get logger
session_log_dir = f'cache/tracking_singlethread_only/logs/'
os.makedirs(session_log_dir, exist_ok=True)
logger = get_logger(f"{video_filepath.split('/')[-1].split('.')[0]}", logdir=session_log_dir)

# initialize session config
session_frames_output_dir = f'cache/tracking_only/output/{video_id}'
if os.path.exists(session_frames_output_dir):
    if os.path.exists(f"{session_frames_output_dir}/end.pb"):
        logger.info(f"session tracking output dir exists with end file, Delete it to rerun the session.")
        sys.exit(0)
    else:
        logger.info(
            f"session tracking output dir exists but no end file, Deleting it and rerunning the session.")
        shutil.rmtree(session_frames_output_dir)
os.makedirs(session_frames_output_dir, exist_ok=True)
t_start_session = time.time()
# setup networks for tracking handlers

# -------Tracking Model init-------
tracking_config = {
    # trackingHandler
    'track_config': f'configs/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half.py',
    'track_checkpoint': f'models/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth',
}
tracking_model = init_tracking_model(tracking_config['track_config'],
                                     tracking_config['track_checkpoint'],
                                     device=DEVICE)

# start loop with frames and video handler
class_video_file = video_filepath
if not os.path.exists(class_video_file):
    logger.info(f"Video File {class_video_file} not available, skipping session...")
    sys.exit(0)

mmcv_video_frames = mmcv.VideoReader(class_video_file)
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
        track_bboxes = track_results['track_bboxes'][0]
        track_results = [dict(bbox=x[1:], track_id=x[0]) for x in list(track_bboxes)]
        track_process_end = datetime.now()
        logger.info(
            f"Frame: {frame_number} | track | {time_diff(track_process_start, track_process_end):.3f} secs")
        # output frame in tracking only dir
        pickle.dump((frame_number, track_results),
                    open(f'{session_frames_output_dir}/{frame_number}.pb', 'wb'))
pickle.dump((frame_number, track_results), open(f'{session_frames_output_dir}/end.pb', 'wb'))
del tracking_model, mmcv_video_frames, logger
torch.cuda.empty_cache()
