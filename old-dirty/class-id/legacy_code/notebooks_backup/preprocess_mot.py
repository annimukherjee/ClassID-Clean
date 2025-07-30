import numpy as np
import pandas as pd
import scipy.stats

pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt, rcParams
# import cv2
import seaborn as sns

sns.set(style="white", context="paper")
from cycler import cycler
import os, sys
import glob
from datetime import datetime, timedelta
from itertools import combinations, product
import base64
from PIL import Image
from io import BytesIO as _BytesIO
import requests
import json
import pickle
from datetime import datetime
from IPython.display import display, Markdown, Latex
from sklearn.metrics import *
import collections
from copy import deepcopy
import traceback
from sympy import Point, Polygon
from decorators import *
from smartprint import smartprint as sprint
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from utils import time_diff, get_logger
# import plotly
# from pandas_profiling import ProfileReport

pd.options.display.max_columns = None

def printm(s): return display(Markdown(s))


def is_overlapping_metric(bu, bv, eps_fraction=0.1):
    X_TL1, Y_TL1, X_BR1, Y_BR1 = bu[:4]
    X_TL2, Y_TL2, X_BR2, Y_BR2 = bv[:4]
    eps_distance = min(X_BR1 - X_TL1, X_BR2 - X_TL2, Y_BR1 - Y_TL1, Y_BR2 - Y_TL2) * eps_fraction
    # if rectangle has area 0, no overlap
    if X_TL1 == X_BR1 or Y_TL1 == Y_BR1 or X_TL2 == X_BR2 or Y_TL2 == Y_BR2:
        return False

    # If one rectangle is on left side of other
    if X_TL1 > X_BR2 - eps_distance or X_TL2 > X_BR1 - eps_distance:
        return False

    # If one rectangle is above other
    if Y_TL1 > Y_BR2 - eps_distance or Y_TL2 > Y_BR1 - eps_distance:
        return False

    return True

CLU_EPS = 0.4
CLU_MIN_PTS = 100
MATCH_DISTANCE_THRESHOLD = 0.2
BBOX_OVERLAP_THRESHOLD = 0.8



video_filepath = sys.argv[1]
video_id = video_filepath.split('/')[-1].split('.')[0]
session_log_dir = f'cache/tracking_singlethread_only/logs/'
os.makedirs(session_log_dir, exist_ok=True)
logger = get_logger(f"{video_id}", logdir=session_log_dir)


tracking_cache_dir = f'cache/tracking_only/output/{video_id}'
session_tracking_cache_dir = f'cache/tracking_only/session/'
postprocessed_id_map_data_dir = f'cache/tracking_only/postprocessed_id_map'
vision_cache_dir = f'cache/vision_only/output/{video_id}'
embedding_cache_dir = f'cache/embedding_output'

# extract frame data
frame_file_data = {}
frame_files = glob.glob(f"{tracking_cache_dir}/*")
frame_file_names = [xr.split("/")[-1] for xr in frame_files]
if 'end.pb' in frame_file_names:
    frame_file_data['is_completed']=True
else:
    frame_file_data['is_completed']=False
frame_ids = [int(xr.split(".")[0]) for xr in frame_file_names if not (xr=='end.pb')]
frame_file_data['frame_ids'] = sorted(frame_ids)
frame_file_data['dir_location'] = tracking_cache_dir

# postprocess tracking data
session_tracking_cache_file = f"{session_tracking_cache_dir}/{video_id}.pb"
if not os.path.exists(session_tracking_cache_file):
    session_dir = frame_file_data['dir_location']
    frame_ids = frame_file_data['frame_ids']
    session_tracking_ids = {}
    for frame_id in frame_ids:
        frame_data = pickle.load(open(f'{session_dir}/{frame_id}.pb', 'rb'))
        frame_tracking_ids = [xr['track_id'] for xr in frame_data[1]]
        # print(frame_id, frame_tracking_ids)
        session_tracking_ids[frame_id] = {int(xr): 1 for xr in frame_tracking_ids}
    df_session_ids = pd.DataFrame.from_dict(session_tracking_ids)
    pickle.dump(df_session_ids, open(session_tracking_cache_file, 'wb'))

session_preprocessed_id_map_file = f"{postprocessed_id_map_data_dir}/{video_id}.pb"
if not os.path.exists(session_preprocessed_id_map_file):
    session_frame_dir = f'{tracking_cache_dir}'
    if not os.path.exists(session_tracking_cache_file):
        printm(f"## Tracking file does not exists for session {video_id}, skipping id-matching...")
        sys.exit(0)

    df_tracking = pickle.load(open(session_tracking_cache_file, "rb")).transpose()
    printm(f'## Raw tracking shape:{df_tracking.shape}')
    printm(f'## Filter non-persistentids')
    MIN_ID_FRAMES = 900  # number of frames an id needs to be a persistent id
    col_start_stop_idxs = []
    for col in df_tracking.columns:
        one_idxs = df_tracking.index[np.where(df_tracking[col] == 1)[0]].values
        col_start_stop_idxs.append([col, one_idxs.min(), one_idxs.max()])
    df_id_start_stop = pd.DataFrame(col_start_stop_idxs, columns=['id', 'min_idx', 'max_idx'])
    df_id_start_stop['total_idxs'] = df_id_start_stop['max_idx'] - df_id_start_stop['min_idx']
    nonpersistent_ids_removed = df_id_start_stop[df_id_start_stop.total_idxs <= MIN_ID_FRAMES]['id'].values
    printm(f'### Total ids before filtering: {df_id_start_stop.shape[0]}')
    df_id_start_stop = df_id_start_stop[df_id_start_stop.total_idxs > MIN_ID_FRAMES].reset_index(drop=True)
    printm(f'### Total ids after filtering: {df_id_start_stop.shape[0]}')

    printm(f'## Map ids into one based on bbox overlap and id start/stop distance')
    MAX_ID_DISTANCE = 900
    MAX_BBOX_OVERLAP = 0.4

    potential_id_maps = {}
    num_possible_maps = 0
    for row_idx, row in df_id_start_stop.iterrows():
        row_maxidx = row['max_idx']

        # get polygon for given id
        id_max_frame = row_maxidx
        id_frame_data = pickle.load(open(f"{session_frame_dir}/{id_max_frame}.pb", "rb"))[1]
        id_frame_data = [xr for xr in id_frame_data if (xr['track_id'] == row['id'])][0]
        id_bb = id_frame_data['bbox'][:4].astype(int)
        X_TL1, Y_TL1, X_BR1, Y_BR1 = id_bb
        p1, p2, p3, p4 = map(Point, [[X_TL1, Y_TL1], [X_TL1, Y_BR1], [X_BR1, Y_BR1], [X_BR1, Y_TL1]])
        id_polygon = Polygon(p1, p2, p3, p4)

        potential_id_matches = df_id_start_stop[(df_id_start_stop.min_idx <= row_maxidx + MAX_ID_DISTANCE) & (
                    df_id_start_stop.min_idx > row_maxidx - MAX_ID_DISTANCE)].id.values
        successful_matches = []
        if len(potential_id_matches) > 0:
            num_possible_maps += 1
            # print('\n',row['id'], potential_id_matches, row['min_idx'],row['max_idx'],row['total_idxs'])
            for matched_id in potential_id_matches:
                matched_id_min_frame = df_id_start_stop[df_id_start_stop.id == matched_id].min_idx.values[0]
                matched_id_frame_data = pickle.load(open(f"{session_frame_dir}/{matched_id_min_frame}.pb", "rb"))[1]
                matched_id_frame_data = [xr for xr in matched_id_frame_data if (xr['track_id'] == matched_id)][0]
                matched_id_bb = matched_id_frame_data['bbox'][:4].astype(int)
                X_TL2, Y_TL2, X_BR2, Y_BR2 = matched_id_bb

                p1, p2, p3, p4 = map(Point, [[X_TL2, Y_TL2], [X_TL2, Y_BR2], [X_BR2, Y_BR2], [X_BR2, Y_TL2]])
                matched_id_polygon = Polygon(p1, p2, p3, p4)

                # find intersection of two polygons
                # check if intersection exists
                if id_polygon.encloses_point(matched_id_polygon.centroid) & matched_id_polygon.encloses_point(
                        id_polygon.centroid):
                    X_TL_in, X_BR_in = sorted([X_TL1, X_TL2, X_BR1, X_BR2])[1:3]
                    Y_TL_in, Y_BR_in = sorted([Y_TL1, Y_TL2, Y_BR1, Y_BR2])[1:3]
                    p1, p2, p3, p4 = map(Point, [[X_TL_in, Y_TL_in], [X_TL_in, Y_BR_in], [X_BR_in, Y_BR_in],
                                                 [X_BR_in, Y_TL_in]])
                    intersection = Polygon(p1, p2, p3, p4)

                    # find polygon overlap
                    area_intersection = np.abs(intersection.area)
                    area_union = np.abs(id_polygon.area) + np.abs(matched_id_polygon.area) - area_intersection
                    overlap_fraction = (area_intersection / area_union).evalf()
                else:
                    overlap_fraction = 0.
                if overlap_fraction > MAX_BBOX_OVERLAP:
                    successful_matches.append((matched_id, overlap_fraction))

                # print('\tMatching Id: ', matched_id,':', 'frame:',matched_id_min_frame,'overlap_fraction:', overlap_fraction)
        if len(successful_matches) > 0:
            successful_matched_id = sorted(successful_matches, key=lambda x: x[1])[-1][0]
            # print(row['id'], '-->Successful match to-->',successful_matched_id)
            if row['id'] in potential_id_maps.keys():
                potential_id_maps[successful_matched_id] = potential_id_maps[row['id']]
            else:
                potential_id_maps[successful_matched_id] = row['id']

    matched_ids = list(potential_id_maps.keys())
    df_id_start_stop = df_id_start_stop[~df_id_start_stop['id'].isin(matched_ids)].sort_values(by='id').reset_index(
        drop=True)
    printm(f'### Total ids after mapping: {df_id_start_stop.shape[0]}')

    printm(f'## Assign new ids to final set of postprocessed ids')
    new_to_old_id_map = df_id_start_stop['id'].to_dict()
    old_to_new_id_map = {v: k for k, v in new_to_old_id_map.items()}

    for matched_id in matched_ids:
        old_to_new_id_map[matched_id] = old_to_new_id_map[potential_id_maps[matched_id]]

    for removed_id in nonpersistent_ids_removed:
        old_to_new_id_map[removed_id] = 10000

    pickle.dump(old_to_new_id_map, open(session_preprocessed_id_map_file, "wb"))
else:
    printm(f"## FILE EXISTS: ID Map for session: {video_id}")
