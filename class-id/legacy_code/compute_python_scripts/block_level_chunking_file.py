'''
This is new edusense pipeline, given a list of session videos and corresponding tracking data location, we generate face, gaze and embedding information.
'''

import os
import sys

from configs.get_session_config import get_session_config
from utils_computev3 import get_logger, time_diff
import mmcv
from datetime import datetime
from utils_computev3 import time_diff, get_logger
from copy import deepcopy
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
import cv2
import shutil
import traceback
# cd /home/prasoon/openmmlab/edusenseV2compute/compute/videoV3/ && conda activate edusense && export PYTHONPATH="${PYTHONPATH}:/home/prasoon/openmmlab/edusenseV2compute/compute/videoV3/"
session_filter_list = [
 'classinsight-cmu_05681A_ghc_4301_201905011630',
 'classinsight-cmu_05681A_ghc_4301_201904171630',
 'classinsight-cmu_05681A_ghc_4301_201902201630',
 'classinsight-cmu_05681A_ghc_4301_201904101630',
 'classinsight-cmu_05681A_ghc_4301_201901231630',
 'classinsight-cmu_05418A_ghc_4102_201902251200',
 'classinsight-cmu_05418A_ghc_4102_201904081200',
 'classinsight-cmu_05418A_ghc_4102_201905011200',
 'classinsight-cmu_05418A_ghc_4102_201904291200',
 'classinsight-cmu_05418A_ghc_4102_201904011200',
 'classinsight-cmu_05748A_ghc_4101_201902051630',
 'classinsight-cmu_05748A_ghc_4101_201902141630',
 'classinsight-cmu_05748A_ghc_4101_201904021630',
 'classinsight-cmu_05748A_ghc_4101_201902281630',
 'classinsight-cmu_05748A_ghc_4101_201903071630',
 'classinsight-cmu_21127J_ghc_4102_201904230930',
 'classinsight-cmu_21127J_ghc_4102_201903260930',
 'classinsight-cmu_21127J_ghc_4102_201904160930',
 'classinsight-cmu_21127J_ghc_4102_201904300930',
 'classinsight-cmu_21127J_ghc_4102_201903190930',
 'classinsight-cmu_05410A_ghc_4301_201904151500',
 'classinsight-cmu_05410A_ghc_4301_201902251500',
 'classinsight-cmu_05410A_ghc_4301_201904081500',
 'classinsight-cmu_05410A_ghc_4301_201904221500',
 'classinsight-cmu_05410A_ghc_4301_201902181500',
                       
 'classinsight-cmu_17214B_ph_a21_201902271030',
 'classinsight-cmu_17214B_ph_a21_201903061030',
 'classinsight-cmu_17214B_ph_a21_201904031030',
 'classinsight-cmu_17214B_ph_a21_201904101030',
 'classinsight-cmu_17214B_ph_a21_201904241030',
 'classinsight-cmu_17214C_ph_225b_201903201130',   
 # 'classinsight-cmu_17214C_ph_225b_201904031130',
 'classinsight-cmu_17214C_ph_225b_201904101130',
 'classinsight-cmu_17214C_ph_225b_201904171130',
 'classinsight-cmu_17214C_ph_225b_201904241130',
 'classinsight-cmu_17214C_ph_225b_201905011130',
 'classinsight-cmu_05410B_ghc_4211_201902111500',
 'classinsight-cmu_05410B_ghc_4211_201903181500',
 'classinsight-cmu_05410B_ghc_4211_201904081500',
 'classinsight-cmu_05410B_ghc_4211_201904151500',
 'classinsight-cmu_05410B_ghc_4211_201904221500',
 'classinsight-cmu_05410B_ghc_4211_201901281500',
]

SOURCE_DIR = '/home/prasoon/openmmlab/edusenseV2compute/compute/videoV3'
VIDEO_DIR = f'/mnt/ci-nas-classes/classinsight/{sys.argv[1]}/video_backup'
COURSE_ID = sys.argv[2]
NUM_SECS_PER_BLOCK = 120
session_dirs = glob.glob(f'{VIDEO_DIR}/*')
TRACK_INFO_DIR = f'/mnt/ci-nas-cache/edulyzeV2/track/{COURSE_ID}'
print(f"Got {len(session_dirs)} sessions from {VIDEO_DIR}.")
session_dirs = [xr for xr in session_dirs if f'_{COURSE_ID}_' in xr]
# session_dirs = [xr for xr in session_dirs if (xr.split("/")[-1] in session_filter_list)]
print(f"Got {len(session_dirs)} sessions from {VIDEO_DIR} and course {COURSE_ID}")


if __name__ == '__main__':
    for SESSION_DIR in session_dirs:
        SESSION_KEYWORD = SESSION_DIR.split("/")[-1]
        for SESSION_CAMERA in ['front','back']:
            SESSION_CAMERA_FILES = glob.glob(f'{SESSION_DIR}/*{SESSION_CAMERA}.avi')
            for SESSION_CAMERA_FILE in SESSION_CAMERA_FILES:
                SESSION_KEYWORD = \
                SESSION_CAMERA_FILE.split("/")[-1].split(f"_{SESSION_CAMERA}.avi")[0].split(f"-{SESSION_CAMERA}.avi")[0]
                
                # get logger
                session_log_dir = f'cache/logs_block_level_chunking/{COURSE_ID}'
                os.makedirs(session_log_dir, exist_ok=True)
                logger = get_logger(f"{SESSION_KEYWORD}-{SESSION_CAMERA}", logdir=session_log_dir)
                logger.info(f"processing for session {SESSION_KEYWORD}, {SESSION_CAMERA}")

                # initialize session config
                session_output_dir = f'/mnt/ci-nas-cache/copus_video_data/{COURSE_ID}/{SESSION_KEYWORD}-{SESSION_CAMERA}'
                if os.path.exists(session_output_dir):
                    if os.path.exists(f"{session_output_dir}/end.pb"):
                        logger.info(f"session output dir exists with end file, Delete it to rerun the session.")
                        continue
                    else:
                        logger.info(
                            f"session tracking output dir exists but no end file, trying to resume.")
                        # shutil.rmtree(session_frames_output_dir)
                os.makedirs(session_output_dir, exist_ok=True)
                t_start_session = datetime.now()

                # start loop with frames and video handler
                # class_video_file = f'{SESSION_DIR}/{SESSION_KEYWORD}-{SESSION_CAMERA}.avi'
                class_video_file = SESSION_CAMERA_FILE
                if not os.path.exists(class_video_file):
                    logger.info(f"Video File {class_video_file} not available, skipping session...")
                    continue
                    
                mmcv_video = mmcv.VideoReader(class_video_file)
                total_frames = mmcv_video.frame_cnt
                video_fps = mmcv_video.fps
                total_duration = total_frames / video_fps
                num_blocks = int(total_duration / NUM_SECS_PER_BLOCK)
                logger.info(f"Total Duration: {total_frames} frames({video_fps} FPS) | {total_duration} secs | {num_blocks} blocks.")
                for block_id in range(num_blocks):
                    block_start, block_end = block_id*NUM_SECS_PER_BLOCK, (block_id+1)*NUM_SECS_PER_BLOCK
                    block_file_path_4k = f'{session_output_dir}/4K_{block_id}.mp4'
                    block_chunk_start = datetime.now()
                    try:
                        mmcv.cut_video(class_video_file, block_file_path_4k, start=block_start, end=block_end, vcodec='h264', log_level='panic')
                        block_chunk_end = datetime.now()
                        logger.info(
                            f"Block: {block_id} | CHUNK | {time_diff(block_chunk_start, block_chunk_end):.3f} secs")
                    except Exception as e:
                        logger.error(f"Error in chunking block {block_id} from session {SESSION_KEYWORD}, camera {SESSION_CAMERA}.")
                        logger.error(traceback.format_exc())

                    block_file_path_final = f'{session_output_dir}/block_{block_id}.mp4'                    
                    if os.path.exists(block_file_path_4k):
                        block_resize_start = datetime.now()
                        try:
                            mmcv.resize_video(block_file_path_4k, block_file_path_final, size=(960,540), log_level='panic')
                            block_resize_end = datetime.now()
                            logger.info(
                                f"Block: {block_id} | RESIZE | {time_diff(block_resize_start, block_resize_end):.3f} secs")
                            os.remove(block_file_path_4k)
                        except Exception as e:
                            logger.error(f"Error in resizing block {block_id} from session {SESSION_KEYWORD}, camera {SESSION_CAMERA}.")
                            logger.error(traceback.format_exc())
                            
                t_end_session = datetime.now()
                pickle.dump((time_diff(t_start_session, t_end_session), num_blocks), open(f'{session_output_dir}/end.pb', 'wb'))
                time.sleep(5)
