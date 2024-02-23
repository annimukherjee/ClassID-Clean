"""
This is main file to run Edusense pipeline with compute V3 on multiprocessing
"""
import os

from configs.get_session_config import get_session_config
from utils_computev3 import get_logger, time_diff
from multiprocessing import Queue, Process
import handlers
import torch


from queue import Empty as EmptyQueueException
import time
import pickle
import glob

SOURCE_DIR = '/home/prasoon/openmmlab/edusenseV2compute/compute/videoV3'
VIDEO_DIR = '/mnt/ci-nas-classes/classinsight/2019F/video_backup'
COURSE_ID = '79388A'
OUT_DIR = '/home/prasoon/video_analysis/edusenseV2compute/compute/videoV3/cache/video'
BACKFILL_STATUS_DIR = '/home/prasoon/video_analysis/edusenseV2compute/compute/videoV3/cache/backfill_status'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BACKFILL_STATUS_DIR, exist_ok=True)
# SESSION_DIR = '/mnt/ci-nas-classes/classinsight/2019F/video_backup'
# SESSION_KEYWORD = 'classinsight-cmu_79388A_ghc_4301_201910031826'
# SESSION_CAMERA = 'front'
# SOURCE_DIR = '/home/prasoon/video_analysis/edusenseV2compute/compute/videoV3'
# SESSION_DIR = '/home/prasoon/video_analysis/mmtracking/'
# SESSION_KEYWORD = 'first-10-min_5fps.mp4'

DEVICE = 'cuda:2'

NUM_FACE_DETECTION_HANDLERS = 1
TARGET_FPS = 3
# START_FRAME_NUMBER = 0 # used for debug purposes only
FRAME_INTERVAL_IN_SEC = 0.6
MAX_QUEUE_SIZE = 300
session_dirs = glob.glob(f'{VIDEO_DIR}/*')
session_dirs = [xr for xr in session_dirs if f'_{COURSE_ID}_' in xr]
print(f"Got {len(session_dirs)} sessions from {VIDEO_DIR} and course {COURSE_ID}")

if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn')
    except:
        print("Spawn method already set, continue...")
    for SESSION_DIR in session_dirs:
        SESSION_KEYWORD = SESSION_DIR.split("/")[-1]
        for SESSION_CAMERA in ['front']:
            logger = get_logger(f"MAIN-{SESSION_KEYWORD}-{SESSION_CAMERA}", logdir=f'cache/logs/{COURSE_ID}')
            logger.info(f"processing for session {SESSION_KEYWORD}, {SESSION_CAMERA}")
            t_start_session = time.time()
            session_config = get_session_config(SOURCE_DIR,
                                                COURSE_ID,
                                                SESSION_DIR,
                                                SESSION_KEYWORD,
                                                SESSION_CAMERA,
                                                DEVICE,
                                                TARGET_FPS,
                                                start_frame_number=0,
                                                frame_interval=FRAME_INTERVAL_IN_SEC)

            session_config['num_face_detection_handlers'] = NUM_FACE_DETECTION_HANDLERS

            # Check if session camera output file exists in output dir
            session_keyword, session_camera = session_config["session_keyword"], session_config["session_camera"]

            session_backfill_status_file = f'{BACKFILL_STATUS_DIR}/{session_keyword}-{session_camera}.txt'
            backfill_completion_status = 'backfill_complete...'
            if os.path.exists(session_backfill_status_file):
                with open(session_backfill_status_file, 'r') as f:
                    most_recent_stats = f.readlines()[-1]
                    if most_recent_stats == backfill_completion_status:
                        logger.info(f"Session backfilled already. Skipping..")
                        continue
                    else:
                        logger.info(f"Session backfill status partial...")
                logger.info(f"Session starting fresh backfilling...")
                os.remove(session_backfill_status_file)

            # Setup multiprocessing queues <input>_<output>_queue
            '''                        
            Queue-Process Architecture:
                                                            
            P:video_handler
                   |
                  \|/                                   |--Q{tracking_pose_queue}--> P:pose_handler ------------>|
            Q{video_tracking_queue}-> P:tracking_handler|                                                        |
                                                        |--Q{tracking_face_queue}-->P:face_handler               |
                                                                                         |                       |--Q{video_output_queue}-->
                                                                                        \|/                      |
                                                                                          --Q{face_gaze_queue}-->|
                                                                                          --Q{face_emb_queue}--->|
            
            '''

            video_tracking_queue = Queue(maxsize=MAX_QUEUE_SIZE)

            tracking_pose_queue = Queue(maxsize=MAX_QUEUE_SIZE)
            tracking_face_queue = Queue(maxsize=MAX_QUEUE_SIZE)

            face_gaze_queue = Queue(maxsize=MAX_QUEUE_SIZE)
            face_emb_queue = Queue(maxsize=MAX_QUEUE_SIZE)

            video_output_queue = Queue(maxsize=MAX_QUEUE_SIZE)

            # initialize output handler
            # output_handler = Process(target=handlers.run_output_handler,
            #                            args=(video_output_queue, session_config,"output"))
            # output_handler.start()

            # initialize support handlers

            # ---------------- Student/Instructor Tracking Handler ----------------
            tracking_handler = Process(target=handlers.run_tracking_handler,
                                       args=(video_tracking_queue, tracking_pose_queue, tracking_face_queue, session_config,
                                             "tracker"))
            tracking_handler.start()

            # ---------------- Student/Instructor Pose Handler ----------------
            pose_handler = Process(target=handlers.run_pose_handler,
                                   args=(tracking_pose_queue, video_output_queue, session_config, "pose"))
            pose_handler.start()

            # ---------------- Student/Instructor Face(Bounding Box) Detection Handler ----------------
            face_detection_handlers = [None] * NUM_FACE_DETECTION_HANDLERS
            for i in range(NUM_FACE_DETECTION_HANDLERS):
                face_detection_handlers[i] = Process(target=handlers.run_face_handler,
                                                     args=(tracking_face_queue, face_gaze_queue, face_emb_queue,
                                                           session_config, "face_detection"))
                face_detection_handlers[i].start()

            # ---------------- Student/Instructor Gaze Detection Handler ----------------
            gaze_handler = Process(target=handlers.run_gaze_handler,
                                   args=(face_gaze_queue, video_output_queue, session_config, "gaze"))
            gaze_handler.start()

            # ---------------- Student/Instructor Face(Dense Embedding) Detection Handler ----------------
            face_embedding_handler = Process(target=handlers.run_face_embedding_handler,
                                             args=(
                                                 face_emb_queue, video_output_queue, session_config,
                                                 "face_embedding"))
            face_embedding_handler.start()

            # ---------------- Student/Instructor Video Processing Handler ----------------
            video_handler = Process(target=handlers.run_video_handler,
                                    args=(None, video_tracking_queue, session_config, "video"))
            video_handler.start()

            # ---------------- All Handlers Initialized ----------------
            session_video_output = {
                'gaze': [],
                'face_embedding': [],
                'pose': []
            }

            final_packet_received = {'gaze': False, 'face_embedding': False, 'pose': False}
            output_start_time = time.time()
            while True:
                try:
                    frame_number, frame_type, frame_data = video_output_queue.get(timeout=0.1)
                except EmptyQueueException:
                    time.sleep(0.5)
                    continue
                if frame_type not in session_video_output.keys():
                    logger.info(f"Frame Type {frame_type} is not supported..")
                    continue
                if frame_data is None:
                    final_packet_received[frame_type] = True
                    if (final_packet_received['gaze']) & (final_packet_received['pose']) & (
                            final_packet_received['face_embedding']):
                        logger.info("All frames received, closing output handler...")
                        session_output_file = f'{OUT_DIR}/{session_keyword}-{session_camera}.pb'
                        pickle.dump(session_video_output, open(session_output_file, 'wb'))
                        with open(session_backfill_status_file, 'a+') as f:
                            f.write(backfill_completion_status)
                        break
                else:
                    session_video_output[frame_type].append((frame_number, frame_data))
                if (frame_number % 1000 == 0) & (frame_type == 'face_embedding'):
                    with open(session_backfill_status_file, 'a+') as f:
                        f.write(f'{frame_number}-{time.time() - output_start_time:3f}')
                    logger.info(f"Stored frames till {frame_number} in {time.time() - output_start_time:3f} secs..")
                    output_start_time = time.time()

            # joining all processes
            video_handler.join()
            tracking_handler.join()
            pose_handler.join()
            for i in range(NUM_FACE_DETECTION_HANDLERS):
                face_detection_handlers[i].join()
            gaze_handler.join()
            face_embedding_handler.join()

            # free resources
            del video_output_queue, tracking_face_queue, tracking_pose_queue, \
                face_emb_queue, face_gaze_queue, video_output_queue
