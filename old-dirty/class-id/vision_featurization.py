
import os
import sys
from copy import deepcopy

from class_mot import session_frames_output_dir
from configs.get_session_config import get_session_config
from utils import get_logger, time_diff
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
import traceback
from torchreid.utils import FeatureExtractor as REIDFeatureExtractor
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image

video_filepath = sys.argv[1]
video_id = video_filepath.split('/')[-1].split('.')[0]
# get logger
session_log_dir = f'cache/tracking_singlethread_only/logs/'
os.makedirs(session_log_dir, exist_ok=True)
logger = get_logger(f"{video_id}", logdir=session_log_dir)


SOURCE_DIR = './'
COURSE_ID = sys.argv[2]
DEVICE_ARG = int(sys.argv[3])
DEVICE_FACE_ARG = int(sys.argv[4])
DEVICE = f'cuda:{DEVICE_ARG}'
DEVICE_FACE = f'cuda:{DEVICE_FACE_ARG}'
TARGET_FPS = 5
FRAME_INTERVAL_IN_SEC = 0.5
FACE_NUM_POOL_WORKERS = 5
session_frames_output_dir = f'cache/vision/output/{video_id}'
# initialize NN Models
session_config = {
        # faceEmbeddingHandler
        'face_embedding_model_name':'vggface2',
        'device': DEVICE
}
# -------pose Model init-------
pose_config = {
    # poseHandler
    'pose_config': f'configs/mmlab/hrnet_w32_coco_256x192.py',
    'pose_checkpoint': f'models/mmlab/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    'kpt_thr': 0.3,
}
pose_model = init_pose_model(pose_config['pose_config'], pose_config['pose_checkpoint'],
                             DEVICE)
pose_dataset = pose_model.cfg.data['test']['type']
pose_dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
config_pose = mmcv.Config.fromfile(pose_config['pose_config'])

# -------face Model init-------
face_model = RetinaFaceInference(device=torch.device(DEVICE_FACE))
face_threadExecutor = ThreadPoolExecutor(FACE_NUM_POOL_WORKERS)
body_count = 0

# -------gaze and embedding Model init-------
gaze_model = GazeInference(device=DEVICE)
facial_embedding_model = InceptionResnetV1(pretrained=session_config['face_embedding_model_name'],
                                           device=DEVICE).eval()

# -------REID extractor init-------
reid_extractor = REIDFeatureExtractor(model_name='osnet_x1_0',device=DEVICE)

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
        track_file = f'cache/tracking_only/output/{video_id}/{frame_number}.pb'
        track_results = pickle.load(open(track_file, 'rb'))[1]
        track_process_end = datetime.now()
        logger.info(
            f"Frame: {frame_number} | track | {time_diff(track_process_start, track_process_end):.3f} secs")

        # get pose output
        pose_process_start = datetime.now()
        h, w, _ = video_frame.shape
        for component in config_pose.data.test.pipeline:
            if component['type'] == 'PoseNormalize':
                component['mean'] = (w // 2, h // 2, .5)
                component['max_value'] = (w, h, 1.)

        process_start = datetime.now()
        if track_results is not None:
            frame_results, _ = inference_top_down_pose_model(pose_model,
                                                             video_frame,
                                                             track_results,
                                                             format='xyxy',
                                                             dataset=pose_dataset,
                                                             dataset_info=pose_dataset_info)
        else:
            frame_results = None
        pose_process_end = datetime.now()
        logger.info(f"Frame: {frame_number} | pose | {time_diff(pose_process_start, pose_process_end):.3f} secs")

        # get face output
        face_process_start = datetime.now()

        if frame_results is not None:
            # face_results = deepcopy(frame_results)
            body_count = len(frame_results)
            body_frames = []
            body_indexes = []
            for body_index, tracking_info in enumerate(frame_results):
                if type(tracking_info) == dict:
                    body_bbox = tracking_info['bbox']
                    X_TL, Y_TL, X_BR, Y_BR = body_bbox[:4].astype(int)
                    if ((Y_BR - Y_TL) < 5) | ((X_BR - X_TL) < 5):
                        logger.warning("Very small body space found, not running face detection...")
                        continue

                    body_frame = video_frame[Y_TL:Y_BR, X_TL:X_BR, :]
                    body_frames.append(body_frame)
                    body_indexes.append(body_index)

            face_detections = face_threadExecutor.map(face_model.run, body_frames)
            for body_index, face_result in zip(body_indexes, face_detections):
                frame_results[body_index].update({
                    'face': face_result[0]
                })

        face_process_end = datetime.now()
        logger.info(
            f"Frame: {frame_number} | face | {time_diff(face_process_start, face_process_end):.3f} secs [{body_count}]")

        # get gaze output
        gaze_process_start = datetime.now()
        if frame_results is not None:
            for body_index, frame_result in enumerate(frame_results):
                body_bbox = frame_result['bbox']
                # logger.info(f'{frame_number},Body: {body_bbox}')
                faces = deepcopy(frame_result.get('face', np.array([])))
                X_TL, Y_TL, X_BR, Y_BR = body_bbox[:4].astype(int)
                if faces.shape[0] > 0:
                    # logger.info(f'{frame_number},Face: {faces[0]}')
                    faces[0][0] += X_TL
                    faces[0][1] += Y_TL
                    faces[0][2] += X_TL
                    faces[0][3] += Y_TL

                    # Get Gaze
                    pred_gazes, _, points_2d, tvecs = gaze_model.run(video_frame, faces, frame_debug=False)
                    frame_results[body_index].update({
                        'rvec': pred_gazes,
                        'gaze_2d': points_2d,
                        'tvec': tvecs,
                    })
        gaze_process_end = datetime.now()
        logger.info(
            f"Frame: {frame_number} | gaze | {time_diff(gaze_process_start, gaze_process_end):.3f} secs")

        # get reid feature vector
        reid_process_start = datetime.now()
        if frame_results is not None:
            body_count = len(frame_results)
            body_frames = []
            body_indexes = []
            for body_index, tracking_info in enumerate(frame_results):
                if type(tracking_info) == dict:
                    body_bbox = tracking_info['bbox']
                    X_TL, Y_TL, X_BR, Y_BR = body_bbox[:4].astype(int)
                    if ((Y_BR - Y_TL) < 5) | ((X_BR - X_TL) < 5):
                        logger.warning("Very small body space found, not running face detection...")
                        continue
                    if (X_TL < 0) | (Y_TL < 0) | (X_BR < 0) | (Y_BR < 0):
                        # print(frame_idx, body_index, Y_TL,Y_BR, X_TL,X_BR, body_frame.shape)
                        logger.warning("Negative boundaries for bounding boxes, skipping...")
                        continue

                    body_frame = video_frame[Y_TL:Y_BR, X_TL:X_BR, :]
                    body_frames.append(body_frame)
                    body_indexes.append(body_index)

            body_reid_features = reid_extractor(body_frames)
            for body_index, body_reid_embedding in zip(body_indexes, body_reid_features):
                frame_results[body_index].update({
                    'reid_features': body_reid_embedding.detach().cpu().numpy()
                })
        reid_process_end = datetime.now()
        logger.info(
            f"Frame: {frame_number} | reid | {time_diff(reid_process_start, reid_process_end):.3f} secs")

        # get facial embedding output

        emb_process_start = datetime.now()
        if frame_results is not None:
            for body_index, frame_result in enumerate(frame_results):
                body_bbox = frame_result['bbox']
                faces = deepcopy(frame_result.get('face', np.array([])))
                X_TL, Y_TL, X_BR, Y_BR = body_bbox[:4].astype(int)
                face_embedding = None
                if faces.shape[0] > 0:
                    try:
                        faces[0][0] += X_TL
                        faces[0][1] += Y_TL
                        faces[0][2] += X_TL
                        faces[0][3] += Y_TL

                        # Get facial embedding for given face.
                        faces = faces[0][:4].astype(int)
                        face_frame = video_frame[faces[1]:faces[3], faces[0]:faces[2], :]
                        target_size = (244, 244)

                        if face_frame.shape[0] > 0 and face_frame.shape[1] > 0:
                            factor_0 = target_size[0] / face_frame.shape[0]
                            factor_1 = target_size[1] / face_frame.shape[1]
                            factor = min(factor_0, factor_1)

                            dsize = (int(face_frame.shape[1] * factor), int(face_frame.shape[0] * factor))
                            face_frame = cv2.resize(face_frame, dsize)

                            diff_0 = target_size[0] - face_frame.shape[0]
                            diff_1 = target_size[1] - face_frame.shape[1]

                            # Put the base image in the middle of the padded image
                            face_frame = np.pad(
                                face_frame,
                                (
                                    (diff_0 // 2, diff_0 - diff_0 // 2),
                                    (diff_1 // 2, diff_1 - diff_1 // 2),
                                    (0, 0),
                                ),
                                "constant",
                            )
                            # double check: if target image is not still the same size with target.
                            if face_frame.shape[0:2] != target_size:
                                face_frame = cv2.resize(face_frame, target_size)

                            # normalizing the image pixels
                            video_frame_pixels = face_frame.astype(np.float32)  # what this line doing? must?
                            video_frame_pixels /= 255  # normalize input in [0, 1]
                            face_tensor = torch.from_numpy(video_frame_pixels).permute(2, 1, 0).unsqueeze(0).to(
                                session_config['device'])
                            face_embedding = facial_embedding_model(face_tensor)[0].to('cpu').detach().numpy()
                    except:
                        print(f"Error for face for body {body_index}, frame: {frame_number}")
                        print(traceback.format_exc())
                    frame_results[body_index].update({
                        'face_embedding': face_embedding
                    })
        emb_process_end = datetime.now()

        logger.info(
            f"Frame: {frame_number} | embedding | {time_diff(emb_process_start, emb_process_end):.3f} secs")

        # output frame in tracking only dir
        pickle.dump((frame_number, frame_results),
                    open(f'{session_frames_output_dir}/{frame_number}.pb', 'wb'))
pickle.dump((frame_number, frame_results), open(f'{session_frames_output_dir}/end.pb', 'wb'))
torch.cuda.empty_cache()
