# Copyright (c) 2017-2019 Carnegie Mellon University. All rights reserved.
# Use of this source code is governed by BSD 3-clause license.

import argparse
import base64
import datetime
from datetime import timedelta
from datetime import datetime
import json
import queue
import os
import socket
import struct
import sys
import threading
import time
import traceback
import sys
import cv2
import numpy
import requests
import math
import logging
from logging.handlers import WatchedFileHandler
import traceback

import cv2
import re
import pytz
import numpy as np
from datetime import datetime,timedelta

from edusense_scripts.centroidtracker import *
# from edusense_scripts.headpose import *
# from edusense_scripts.render import *
from edusense_scripts.process import *

def process_frame(frame_number, frame_data, centroid_tracker, logger):
    t_process_frame_start = datetime.now()
    logger.info("===Start Processing New Frame===")
    start_time = time.time()

    try:
        json_time = 0
        featurization_time = 0
        thumbnail_time = 0
        interframe_time = 0
        t_json_start = datetime.now()
        if frame_number is not None:
            frame_data['frameNumber'] = frame_number
            print(f"Got frame number %d", frame_number)

        t_json_end = datetime.now()
        json_time = time_diff(t_json_start, t_json_end)
        logger.info("Loaded json data in %.3f secs", json_time)

        image_rows = 0
        image_cols = 0
        has_raw_image = "rawImage" in frame_data.keys()
        has_thumbnail = "thumbnail" in frame_data.keys()
        raw_image = None

        # Featurization
        logger.info("--Start frame featurization--")
        t_featurization_start = datetime.now()
        # featurization_start_time = time.time()

        # extract key points
        bodies = frame_data['people']
        rects = []
        person_poses = []
        logger.info(f"Count of bodies found in frame: {str(len(bodies))}")
        bodies = list(filter(lambda b: check_body_pts(b['body']), bodies))

        logger.info(f"Count of bodies found in area of interest: {str(len(bodies))}")
        t_prepare_interframe_start = datetime.now()

        for body_idx, body in enumerate(bodies):
            body_keypoints = body["body"]
            face_keypoints = body["face"] if "face" in body.keys() else None

            # prune body keypoints
            body_keypoints = prune_body_pts(body_keypoints)
            body["body"] = body_keypoints
            pose = get_pose_pts(body_keypoints)
            body['inference'] = {
                'posture': {},
                'face': {},
                'head': {}
            }
            # prepare inter-frame tracking
            box = get_pose_box(pose)
            rects.append(box.astype("int"))
            person_poses.append(pose)

        t_prepare_interframe_end = datetime.now()
        logger.info("Pruned body kps and prepare for interframe processing in %.3f secs",
                    time_diff(t_prepare_interframe_start, t_prepare_interframe_end))

        # Interframe
        logger.info("Start interframe processing")
        t_run_interframe_start = datetime.now()
        tracking_id = None

        objects, poses = centroid_tracker.update(rects, person_poses)

        for body in bodies:
            body_keypoints = body["body"]
            pose = get_pose_pts(body_keypoints)
            for (objectID, person_pose) in poses.items():
                if pose[1][0] == person_pose[1][0] and pose[1][1] == person_pose[1][1]:
                    body['inference']['trackingId'] = objectID + 1
                    break

        t_run_interframe_end = datetime.now()
        interframe_time = time_diff(t_run_interframe_start, t_run_interframe_end)
        logger.info("Finish interframe processing in %.3f secs", interframe_time)
        # featurization_start_time =time.time()


        body_time_profile = {}
        for body_idx, body in enumerate(bodies):

            body_time_profile[body_idx] = {
                'trackingId': str(body.get('inference', {}).get('trackingId', 'notFound'))
            }  # profile time for each person

            start_time = time.time()
            body_keypoints = body["body"]
            face_keypoints = body["face"] if "face" in body.keys(
            ) else None
            pose = get_pose_pts(body_keypoints)
            body_time_profile[body_idx]['get_keypoints'] = round(time.time() - start_time, 3)

            start_time = time.time()
            # face orientation
            faceOrientation = None
            faceOrientation = get_facing_direction(pose)
            body_time_profile[body_idx]['get_face_orient'] = round(time.time() - start_time, 3)

            # Sit stand
            start_time = time.time()
            sit_stand, color_stand, pts = predict_sit_stand(body_keypoints)
            body_time_profile[body_idx]['get_sit_stand'] = round(time.time() - start_time, 3)

            # Armpose
            start_time = time.time()
            armpose, color_pose, pts = predict_armpose(body_keypoints)
            body_time_profile[body_idx]['get_armpose'] = round(time.time() - start_time, 3)

            # Mouth
            start_time = time.time()
            mouth = None
            smile = None
            if face_keypoints is not None:
                mouth, _, smile, _ = predict_mouth(face_keypoints)

            tvec = None
            yaw = None
            pitch = None
            roll = None
            gaze_vector = None
            face = get_face(pose)
            body_time_profile[body_idx]['get_smile'] = round(time.time() - start_time, 3)

            
            start_time = time.time()
            if armpose is not None:
                body['inference']['posture']['armPose'] = armpose
            if sit_stand is not None:
                body['inference']['posture']['sitStand'] = sit_stand
            if face is not None:
                body['inference']['face']['boundingBox'] = face
            if mouth is not None:
                body['inference']['face']['mouth'] = mouth
            if smile is not None:
                body['inference']['face']['smile'] = smile
            if yaw is not None:
                body['inference']['head']['yaw'] = yaw
            if pitch is not None:
                body['inference']['head']['pitch'] = pitch
            if roll is not None:
                body['inference']['head']['roll'] = roll
            if faceOrientation is not None:
                body['inference']['face']['orientation'] = faceOrientation
            if gaze_vector is not None:
                body['inference']['head']['gazeVector'] = gaze_vector
            if tvec is not None:
                body['inference']['head']['translationVector'] = tvec

            body_time_profile[body_idx]['fill_inferences'] = round(time.time() - start_time, 3)

        
        t_featurization_end = datetime.now()

        featurization_time = time_diff(t_featurization_start, t_featurization_end)
        logger.info("Finish featurization in %.3f secs", featurization_time)


        frame_data['people'] = bodies


    except Exception as e:
        traceback.print_exc(file=sys.stdout)
    t_process_frame_end = datetime.now()
    logger.info("Finished processing frame in %.3f secs", time_diff(t_process_frame_start, t_process_frame_end))
    return frame_data



def get_timestamp(frame):

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cropped_image=gray[80:150,3000:3800]
      cropped_image=cv2.resize(cropped_image,(800,100))
      binary = cv2.adaptiveThreshold(cropped_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,60)
      # text = pytesseract.image_to_string(binary,config='--psm 13  -c tessedit_char_whitelist=:-0123456789APM" " ')
      text = ''
      return text;


def convert24hour(hour,PM_time):
    if  not PM_time and hour==12:
        return 0;
    if  PM_time and hour!=12:
        return hour+12
    else:
        return hour

### NOTE-: if AM and PM is not present in timestamp , by default PM
def clean_OCR_Time(OCR_time):
   ## split the time-date
   split=OCR_time.split(' ')
   ## save time
   time=split[1]
   time_format=''
   ## 0 means PM is not present in the OCR time
   ## 1 means PM is present in the OCR time 
   PM_time=1;
   if len(split)>2 and split[2]=='AM':
       PM_time=0
   
   ## extract unenecessary char 
   for num,ix in enumerate(time):
     if ix.isdigit() or ix==':' :
        time_format=time_format+ix        

   ## use regx to further clean extraction
   p = re.compile('\d{1,2}')
   time_format=p.findall(time_format)
   ## some error with extraction 

   if (len(time_format)<3):
      return None;
   
   ## convert string to datetime with error_handling                    
   hour_OCR=int(time_format[0])
   hour_OCR=convert24hour(hour_OCR,PM_time);
   if hour_OCR>24:
      return None;
   Min_OCR=int(time_format[1])
   if Min_OCR>60:
        return None;
   sec_OCR=int(time_format[2])
   if sec_OCR>60:
        sec_OCR=0;
   time_OCR=timedelta(hours=hour_OCR,minutes=Min_OCR,seconds=sec_OCR)
   return (split[0],time_OCR)

def convert_to_UTC(date,time):

    str_time=str(time)
    dt=date+' '+str_time
    timezone = pytz.timezone('America/New_York')
    try:
       D=datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    except Exception as e :
       try:
         D=datetime.strptime(dt, "%m-%d-%y %H:%M:%S")
       except Exception as e :
         try:
           D=datetime.strptime(dt, "%d-%m-%y %H:%M:%S")
         except Exception as e :
           return(date,time)

    t=timezone.localize(D)
    UTC=t.astimezone(pytz.utc)
    date=str(UTC.date())
    time=timedelta(hours=UTC.hour,minutes=UTC.minute,seconds=UTC.second)
    return(date,time)

def extract_date(video):
   split=video.split('/')
   video_name=''
   for File in split:
      if File.find('.avi') != -1:
        video_name=File
        break;
   name_list=video_name.split('_')
   date_time=''
   for ix in name_list[4]:
      if ix.isdigit():
            date_time=date_time+ix
   year=date_time[:4]
   month=date_time[4:6]
   day=date_time[6:8]
   date=year+'-'+month+'-'+day
   hour=int(date_time[8:10])
   Min=int(date_time[10:12])
   time_delta= timedelta(hours=hour,minutes=Min)
   return (date,time_delta)


def extract_time(video,log):
     
   # print(pytesseract.get_tesseract_version())
   threshold_error=timedelta(hours=1,minutes=0)
   ocr_time_failed=False;
   file_time_failed=False;
   file_name_time=None;
   file_name_date=None;
   fps=None;
   default_time=timedelta(hours=9,minutes=0)
   default_date="2020-05-28"
    
   try: 
     file_name_date,file_name_time=extract_date(video)
     ## check date format
     date_match=datetime.strptime(file_name_date, "%Y-%m-%d")
   except Exception as e:
       log.write("ERROR in extracting the date-time from the file_name\n")
       log.write(str(e)+"\n")
       file_time_failed=True;

   try:
      video_object=cv2.VideoCapture(video)
      print(video)
      fps = video_object.get(cv2.CAP_PROP_FPS)
      ret,frame=video_object.read()
      print(ret)
      ocr_time_stamp=get_timestamp(frame)
      ocr_date,ocr_time=clean_OCR_Time(ocr_time_stamp)
   except Exception as e:
       log.write(video+"ERROR in extracting the date-time from the OCR\n")
       log.write(str(e)+"\n")
       ocr_time_failed=True;

   if(file_time_failed and ocr_time_failed):
       log.write("Using a default time_stamp "+default_date+'T'+str(default_time)+"\n")
       return (fps,default_date,default_time)

   elif ocr_time_failed:
       log.write("Using file extracted time_stamp "+file_name_date+"T"+str(file_name_time)+"\n")
       file_name_date,file_name_time=convert_to_UTC(file_name_date,file_name_time)
       return(fps,file_name_date,file_name_time)
       
   elif file_time_failed:
       log.write("Using OCR extracted time_stamp and OCR date "+ocr_date+"T"+str(ocr_time)+"\n")
       ocr_date,ocr_time=convert_to_UTC(ocr_date,ocr_time)
       return(fps,ocr_date,ocr_time)

   else:
       if abs(ocr_time-file_name_time) < threshold_error:
           log.write("Using OCR timestamp "+file_name_date+"T"+str(ocr_time)+"\n")
           file_name_date,ocr_time=convert_to_UTC(file_name_date,ocr_time)
           return(fps,file_name_date,ocr_time)
       else:
           log.write("Using file_name timestamp "+file_name_date+"T"+str(file_name_time)+"\n")
           file_name_date,file_name_time=convert_to_UTC(file_name_date,file_name_time)
           return(fps,file_name_date,file_name_time)



def time_diff(t_start, t_end):
    """
    Get time diff in secs

    Parameters:
        t_start(datetime)               : Start time
        t_end(datetime)                 : End time

    Returns:
        t_diff(int)                     : time difference in secs
    """

    return (t_end - t_start).seconds + np.round((t_end - t_start).microseconds / 1000000, 3)


