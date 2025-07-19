#!/usr/bin/env python3
"""
_webcam_tracking.py_

A simplified, single-threaded OC-SORT tracking pipeline that reads from your webcam
instead of a directory of AVI files. Results (frame_number + track list) are pickled to
cache/tracking_singlethread_webcam/, and you get a live display with bounding boxes.
"""

import os
import time
import argparse
import pickle
import cv2
import mmcv
from mmtrack.apis import inference_mot, init_model as init_tracking_model
from pprint import pprint

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Where your ClassID / edusenseV2compute repo lives; adjust if needed
SOURCE_DIR = './'

# OC-SORT config & weights (same defaults as before)
TRACK_CONFIG = os.path.join(SOURCE_DIR, 'configs/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half.py')
print(f"[INFO] Using tracking config: {TRACK_CONFIG}")

TRACK_CHECKPOINT = os.path.join(SOURCE_DIR, 'models/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth')
print(f"[INFO] Using tracking checkpoint: {TRACK_CHECKPOINT}")

DEVICE     = 'cpu'   # no CUDA on MacBook Pro
TARGET_FPS = 5       # process at 5 FPS

# Where to dump per-frame pickles
OUTPUT_DIR = 'cache/tracking_singlethread_webcam_understand'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main(camera_index: int):
    # -----------------------------------------------------------------------------
    # 1. Initialize OC-SORT tracking model
    # -----------------------------------------------------------------------------
    print(f"[INFO] Initializing OC-SORT model on {DEVICE} …")
    tracking_model = init_tracking_model(
        TRACK_CONFIG, TRACK_CHECKPOINT, device=DEVICE
    )

    # -----------------------------------------------------------------------------
    # 2. Open the webcam
    # -----------------------------------------------------------------------------
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}")
        return

    # Try to read FPS from camera; default to 30 if unavailable
    print(f"[INFO] Reading camera FPS from {camera_index}…")
    print(f"camera FPS is: {cap.get(cv2.CAP_PROP_FPS)}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Compute how many frames to skip to achieve TARGET_FPS
    num_skip = int(video_fps / TARGET_FPS) if TARGET_FPS > 0 else 0
    print(f"[INFO] Camera FPS={video_fps:.1f}, skipping every {num_skip} frames to run at {TARGET_FPS} FPS")

    frame_number = 0
    last_tracks = []

    # -----------------------------------------------------------------------------
    # 3. Main loop: grab frames, run inference, draw & save results
    # -----------------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Webcam stream ended or error reading frame.")
            break
        print(f"[DEBUG] Processing frame_number: {frame_number}")
        # Only run inference on every Nth frame to match TARGET_FPS
        if num_skip == 0 or frame_number % num_skip == 0:
            t0 = time.time()
            # inference_mot returns e.g. {'track_bboxes': [array([...])], ...}
            mot_results   = inference_mot(tracking_model, frame, frame_id=frame_number)
            print("mot_results keys:", list(mot_results.keys()))
            print("mot_results content:")
            pprint(mot_results)
            
            # mot_results content:
            #     {'det_bboxes': [array([[3.3994254e+02, 3.1487589e+02, 1.8065096e+03, 2.4646357e+03,
            #             7.0425403e-01],
            #         [6.2885602e+02, 2.8017041e+02, 1.9814911e+03, 2.3229517e+03,
            #             3.6180384e-02]], dtype=float32)],
            #     'track_bboxes': [ array([[0.00000000e+00, 3.39942535e+02, 3.14875885e+02, 1.80650964e+03,
            #             2.46463574e+03, 7.04254031e-01]])]}

            raw_bboxes = mot_results['track_bboxes'][0]

            # Reformat to a list of dicts for pickling
            tracks = []
            for det in raw_bboxes:
                track_id = int(det[0])
                x1, y1, x2, y2 = map(int, det[1:5])
                tracks.append({'track_id': track_id, 'bbox': (x1, y1, x2, y2)})

            t1 = time.time()
            print(f"[Frame {frame_number:04d}] {len(tracks)} tracks  |  inference {t1-t0:.3f}s")

            # Draw boxes & IDs
            for tr in tracks:
                x1, y1, x2, y2 = tr['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, str(tr['track_id']),
                            (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Show live result
            cv2.imshow('OC-SORT Webcam Tracking', frame)

            # Save pickle for this frame
            with open(os.path.join(OUTPUT_DIR, f'{frame_number}.pickle'), 'wb') as f:
                pickle.dump((frame_number, tracks), f)
            
            last_tracks = tracks  # keep for end.pickle

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting on user request.")
            break

        frame_number += 1

    # -----------------------------------------------------------------------------
    # 4. Write end marker and cleanup
    # -----------------------------------------------------------------------------
    with open(os.path.join(OUTPUT_DIR, 'end.pickle'), 'wb') as f:
        pickle.dump((frame_number, last_tracks), f)
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Pickles saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Single-threaded OC-SORT tracking from webcam"
    )
    p.add_argument('--camera', type=int, default=0,
                   help="OpenCV camera index (default: 0)")
    args = p.parse_args()
    main(args.camera)
