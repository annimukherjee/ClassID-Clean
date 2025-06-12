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
embmatched_id_map_data_dir = f'cache/embmatched_id_map_data'

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

#   generic loop to get embedding info
session_emb_cache_file = f"{embedding_cache_dir}/{video_id}.pb"
try:
    if not os.path.exists(session_emb_cache_file):
        session_dir = frame_file_data['dir_location']
        frame_ids = frame_file_data['frame_ids']
        session_emb_info = {}
        for frame_id in frame_ids:
            frame_number, frame_data = pickle.load(open(f'{session_dir}/{frame_id}.pb', 'rb'))
            frame_emb_info = {int(person_info['track_id']): {
                'bbox': person_info['bbox'] if 'bbox' in person_info else None,
                'rvec': person_info['rvec'] if 'rvec' in person_info else None,
                'face': person_info['face'] if 'face' in person_info else None,
                'gaze_2d': person_info['gaze_2d'] if 'gaze_2d' in person_info else None,
                'face_embedding': person_info['face_embedding'] if 'face_embedding' in person_info else None,
            } for person_info in frame_data}
            session_emb_info[frame_id] = frame_emb_info
        pickle.dump(session_emb_info, open(session_emb_cache_file, 'wb'))
        print(f"Got emb info for session: {video_id}")
    else:
        print(f"FILE EXISTS: emb info for session: {video_id}")
except Exception as e:
    print(f"Error in getting emb info for session: {video_id}")
    print(e)
    print(traceback.format_exc())
    sys.exit(0)

CLU_EPS = 0.4
CLU_MIN_PTS = 100
MATCH_DISTANCE_THRESHOLD = 0.2
BBOX_OVERLAP_THRESHOLD = 0.8

embmatch_map_cache_file = f"{embmatched_id_map_data_dir}/{video_id}.csv"

try:
    if not os.path.exists(session_emb_cache_file):
        session_emb_info = pickle.load(open(f'{session_emb_cache_file}/{video_id}.pb', 'rb'))
        session_id_map = pickle.load(open(f"{postprocessed_id_map_data_dir}/{video_id}.pb", "rb"))
        df_tracking_new = pickle.load(open(f"{session_tracking_cache_dir}/{video_id}.pb", "rb")).transpose()

        printm("## Replace raw ids with mapped ids for given session")
        session_emb_info = {
            xr: {
                session_id_map[yr]: session_emb_info[xr][yr]
                for yr in session_emb_info[xr] if not (session_id_map[yr] == 10000)} for xr in session_emb_info}

        # arrange info as per new tracking id for entire session
        printm("## arrange info as per new tracking id for entire session")
        gaze_info = {}
        emb_info = {}
        bbox_info = {}

        for frame_number in session_emb_info:
            for trackId in session_emb_info[frame_number]:
                if trackId not in gaze_info:
                    gaze_info[trackId] = []
                    emb_info[trackId] = []
                    bbox_info[trackId] = []
                # get  gaze info
                try:
                    id_bbox = session_emb_info[frame_number][trackId]['bbox']
                    bbox_info[trackId].append([frame_number] + list(id_bbox))
                    pitch, roll, yaw = session_emb_info[frame_number][trackId]['rvec'][0]
                    pitch, roll, yaw = np.rad2deg(pitch), np.rad2deg(roll), np.rad2deg(yaw)
                    gaze_sx, gaze_sy, gaze_ex, gaze_ey = session_emb_info[frame_number][trackId]['gaze_2d'][
                        0].flatten()
                    gaze_info[trackId].append(
                        [frame_number, pitch, roll, yaw, gaze_sx, gaze_sy, gaze_ex, gaze_ey])
                    face_emb = session_emb_info[frame_number][trackId]['face_embedding'].tolist()
                    emb_info[trackId].append([frame_number] + face_emb)
                except:
                    continue

        for id in gaze_info:
            gaze_info[id] = pd.DataFrame(gaze_info[id],
                                         columns=['frame', 'pitch', 'roll', 'yaw', 'gaze_sx', 'gaze_sy',
                                                  'gaze_ex', 'gaze_ey']).set_index('frame')
            emb_info[id] = pd.DataFrame(emb_info[id], columns=['frame'] + np.arange(512).tolist()).set_index(
                'frame')
            bbox_info[id] = pd.DataFrame(bbox_info[id], columns=['frame'] + np.arange(5).tolist()).set_index(
                'frame')

        # Get id start stop for given session (needed to evaluate overlap conditions)
        printm("## Get id start stop for given session (needed to evaluate overlap conditions)")
        total_idxs = df_tracking_new.index.max()
        for old_id in session_id_map:
            new_id = session_id_map[old_id]
            if not new_id == 10000:
                new_id_col = f'N{new_id}'
                if new_id_col not in df_tracking_new:
                    df_tracking_new[new_id_col] = None
                df_tracking_new[new_id_col] = df_tracking_new[new_id_col].where(
                    ~df_tracking_new[new_id_col].isnull(), df_tracking_new[old_id])
            df_tracking_new = df_tracking_new.drop(old_id, axis=1)

        col_start_stop_idxs = []
        for col in df_tracking_new.columns:
            one_idxs = df_tracking_new.index[np.where(df_tracking_new[col] == 1)[0]].values
            col_start_stop_idxs.append([col, one_idxs.min(), one_idxs.max()])
        df_id_start_stop = pd.DataFrame(col_start_stop_idxs, columns=['id', 'min_idx', 'max_idx'])
        df_id_start_stop['total_idxs'] = df_id_start_stop['max_idx'] - df_id_start_stop['min_idx']
        df_id_start_stop['id'] = df_id_start_stop['id'].apply(lambda x: int(x[1:]))

        # Use spectral clustering to get clean set of embeddings and calculate their centroid
        printm("## Use spectral clustering to get clean set of embeddings and calculate their centroid")
        np.random.seed(42)
        clustered_median_emb = {}
        for id in emb_info:
            emb_clu = DBSCAN(min_samples=CLU_MIN_PTS, eps=CLU_EPS)
            try:
                emb_clu.fit(emb_info[id].values)
            except:
                emb_clu = None
            if (emb_clu is None) or (max(emb_clu.labels_) < 0):
                sprint(f"All frames are outliers, not proceeding with id {id}")
                continue
            best_cluster_id = pd.Series(emb_clu.labels_[emb_clu.labels_ >= 0]).value_counts().index[0]
            frames = emb_info[id].iloc[emb_clu.labels_ == best_cluster_id].index.values
            clustered_median_emb[id] = np.median(emb_info[id].loc[frames], axis=0)

        # Evaluate matching distance for temporally non overlapping ids
        printm("## Evaluate matching distance for temporally non overlapping ids")
        match_scores = {}
        for idA in sorted(clustered_median_emb.keys()):
            for idB in sorted(clustered_median_emb.keys()):
                if idB in match_scores.keys():
                    continue
                # check if idA and idB overlaps, if not, Just leave them be
                min_idxA, max_idxA = \
                df_id_start_stop[df_id_start_stop['id'] == idA][['min_idx', 'max_idx']].values[0].tolist()
                min_idxB, max_idxB = \
                df_id_start_stop[df_id_start_stop['id'] == idB][['min_idx', 'max_idx']].values[0].tolist()
                if len(range(max(min_idxA, min_idxB),
                             min(max_idxA, max_idxB))) > 150:  # more than 10 seconds of overlap
                    # overlapping ranges
                    continue
                match_distance = \
                cdist(clustered_median_emb[idA].reshape(1, -1), clustered_median_emb[idB].reshape(1, -1))[0][0]
                if match_distance < MATCH_DISTANCE_THRESHOLD:
                    if idA not in match_scores:
                        match_scores[idA] = {}
                    match_scores[idA][idB] = match_distance

        df_matching_method = pd.DataFrame(match_scores)
        # Evaluate bbox overlap to filter out spatially overlapping ids
        printm("## Evaluate bbox overlap to filter out spatially overlapping ids")
        overlap_scores = {}
        for idA in sorted(clustered_median_emb.keys()):
            for idB in sorted(clustered_median_emb.keys()):
                if idB in match_scores.keys():
                    continue
                # check if idA and idB overlaps, if not, Just leave them be
                min_idxA, max_idxA = \
                df_id_start_stop[df_id_start_stop['id'] == idA][['min_idx', 'max_idx']].values[0].tolist()
                min_idxB, max_idxB = \
                df_id_start_stop[df_id_start_stop['id'] == idB][['min_idx', 'max_idx']].values[0].tolist()
                if len(range(max(min_idxA, min_idxB), min(max_idxA, max_idxB))) > 0:
                    # overlapping ranges
                    continue
                bbox_overlap_matrix = cdist(bbox_info[idA].iloc[:1000], bbox_info[idB].iloc[:1000],
                                            metric=is_overlapping_metric)
                bbox_overlap = np.mean(bbox_overlap_matrix.flatten())
                sprint(idA, idB, bbox_overlap)
                if bbox_overlap > BBOX_OVERLAP_THRESHOLD:
                    if idA not in overlap_scores:
                        overlap_scores[idA] = {}
                    overlap_scores[idA][idB] = bbox_overlap

        df_overlap = pd.DataFrame(overlap_scores)

        # get eligible pairs from matching and spatial overlap information
        printm("## get eligible pairs from matching and spatial overlap information")
        if (df_matching_method.shape[0] == 0) or (df_overlap.shape[0] == 0):
            df_eligible_pairs = pd.DataFrame(columns=["id_pair", "value_overlap", "value_match"])
        else:
            df_overlap_melted = df_overlap.reset_index().melt(id_vars='index')
            df_overlap_melted = df_overlap_melted[~df_overlap_melted['value'].isnull()]
            df_overlap_melted['id_pair'] = df_overlap_melted.apply(
                lambda row: tuple(sorted([int(row['index']), int(row['variable'])])), axis=1)
            df_overlap_melted = df_overlap_melted[['id_pair', 'value']]
            df_overlap_melted

            df_match_melted = df_matching_method.reset_index().melt(id_vars='index')
            df_match_melted = df_match_melted[~df_match_melted['value'].isnull()]
            df_match_melted['id_pair'] = df_match_melted.apply(
                lambda row: tuple(sorted([int(row['index']), int(row['variable'])])), axis=1)
            df_match_melted = df_match_melted[['id_pair', 'value']]
            df_match_melted

            df_eligible_pairs = pd.merge(df_overlap_melted, df_match_melted, on='id_pair',
                                         suffixes=('_overlap', '_match'))
        df_eligible_pairs.to_csv(embmatch_map_cache_file, index=False)

        embmatch_raw_data_dict = {
            'overlap_df': df_overlap,
            'match_df': df_matching_method,
            'eligible_pairs_df': df_eligible_pairs,
            'id_session_embeddings': clustered_median_emb,
            'id_start_stop_df': df_id_start_stop
        }
        pickle.dump(embmatch_raw_data_dict, open(f"{embedding_cache_dir}/{video_id}.pb", "wb"))
        printm(
            f"## Got embedding based id match for session: {video_id}")
        print(f"{df_eligible_pairs.id_pair.values}")
    else:
        print(
            f"FILE EXISTS: embedding based id match for session: {video_id}")
except:
    printm(
        f"# ERROR: Unable to get embedding based id match for: {video_id}")
    print(traceback.format_exc())



