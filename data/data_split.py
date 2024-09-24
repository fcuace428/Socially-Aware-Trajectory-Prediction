import numpy as np
import os
import pandas as pd
import math
import cv2

def choice_data(number_list, dataname, obs_len=10, pred_len=20, skip=1):
    for count, clip_name in enumerate(number_list):
        seq_list_dir = []
        seq_list_bbox = []
        seq_list_speed = []
        seq_list_int = []
        seq_list_imu = []
        seq_list_neigh = []
        current_ann_data, frames_dir = [], []
        seq_len = obs_len + pred_len
        # read data
        img_dir = os.path.join("data/dataset/Titan/images_anonymized/", clip_name, "images")
        img_dir=img_dir.replace('\\', '/')
        ann_dir = os.path.join("data/dataset/Titan/", "titan_0_4")
        imu_dir = os.path.join("data/dataset/Titan/", "imu_data", clip_name)
        ann = pd.read_csv(os.path.join(ann_dir, f"{clip_name}.csv"))
        imu = pd.read_csv(os.path.join(imu_dir, "synced_sensors.csv"), header=None)
        if imu.shape[1] == 6:
            imu.columns = ['img_ts', 'img_path', 'accel_ts', 'accel_ms', 'gyro_ts', 'ang_vel']
        else:
            imu.columns = ['img_ts', 'img_path', 'accel_ts', 'accel_ms', 'gyro_ts', 'ang_vel', 'Nan']
        # get frames number (no repeat)
        frames = np.unique(ann["frames"]).tolist()
        # make data we need
        for i in range(len(ann)):
            ann.iat[i, 0] = '/' + img_dir + '/' + ann.iat[i, 0]
            
            if pd.isna(ann['attributes.Motion Status'][i]):
                p = ann['attributes.Atomic Actions'][i]
                ann['attributes.Motion Status'][i] = p

        ann['nleft'] = np.clip(ann['left'] / 2704, 0, 1)
        ann['ntop'] = np.clip(ann['top'] / 1520, 0, 1)
        ann['nwidth'] = np.clip(ann['width'] / 2704, 0, 1)
        ann['nheight'] = np.clip(ann['height'] / 1520, 0, 1)

        ann['ncenter_x'] = ann['nleft'] + ann['nwidth'] * 0.5
        ann['ncenter_y'] = ann['ntop'] + ann['nheight'] * 0.5
        ann['ndis_x'] = ann['ncenter_x'] - 0.5
        ann['ndis_y'] = ann['ncenter_y'] - 1
        ann['nright'] = ann['nleft'] + ann['nwidth']
        ann['ndown'] = ann['ntop'] + ann['nheight']
        ann_need = ann.reindex(columns=['frames', 'obj_track_id', 'label', 'nleft', 'ntop', 'nright', 'ndown',\
            'ncenter_x', 'ncenter_y', 'nwidth', 'nheight','ndis_x', 'ndis_y', 'attributes.Motion Status'])
        ann_need["label"] = ann_need["label"].map({"person": 0, "vehicle_4_wheeled": 1, "vehicle_2_wheeled": 2})
        ann_need["attributes.Motion Status"] = ann_need["attributes.Motion Status"]\
        .map({"parked":0, "moving":1, "stopped":2, "running":3, "bending":4, "jumping":5,"sitting":6,\
        "walking":7, "laying down":8, "squatting":9, "standing":10, "none of the above":11})
        imu_need = np.array(imu.reindex(columns=['img_path', 'accel_ms', 'ang_vel']))

        for i, img_name in enumerate(frames):
            # add frames to dir
            frames_dir.append('/' + img_dir + '/' + frames[i])
            current_ann_data.append(np.array(ann_need[ann_need["frames"] == '/' + img_dir + '/' + img_name]))
            
        num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))
        for idx in range(0, num_sequences * skip + 1, skip):
            #concate 30 frame(ex: 00006~000180)
            curr_seq_data = np.concatenate(current_ann_data[idx:idx + seq_len], axis=0)
            
            # get id number (no repeat)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            # obj dir
            curr_seq_dir = np.zeros((len(peds_in_curr_seq), seq_len, 1), dtype="S100")
            # obj bbox
            curr_seq_bbox = np.zeros((len(peds_in_curr_seq), seq_len, 11))
            # obj speed
            curr_seq_speed = np.zeros((len(peds_in_curr_seq), seq_len, 8))
            # obj intention
            curr_seq_int = np.zeros((len(peds_in_curr_seq), seq_len, 1))
            # obj imu
            curr_seq_imu = np.zeros((len(peds_in_curr_seq), seq_len, 2))
            # neighbor
            curr_seq_neighbor = np.zeros((len(peds_in_curr_seq), seq_len, 50, 9))
            # num_peds_considered count total ped can use
            num_peds_considered = 0

            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_dir_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, 0]
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, 1::].astype(float)
                pad_front = frames_dir.index(curr_dir_seq[0]) - idx
                pad_end = frames_dir.index(curr_dir_seq[-1]) - idx + 1
                if pad_end - pad_front != seq_len or curr_dir_seq.shape[0] != seq_len:
                    continue
                # make imu
                curr_imu_seq = []
                for i in range(imu_need.shape[0]):
                    a = imu_need[i][0]
                    if curr_dir_seq[0] == '/data/dataset/Titan/images_anonymized/' + a:
                        curr_imu_seq = imu_need[i:i + 30, 1::]
                        break
                # make neighbors
                curr_neighbor_seq = np.zeros((seq_len, 50, 9))
                for i, ann_clip in enumerate(curr_dir_seq):
                    for frame_data in current_ann_data:
                        if ann_clip == frame_data[0, 0]:
                            # bbox
                            o = frame_data[:, 2:-3]
                            curr_neighbor_seq[i, :o.shape[0], ::] = o
                # make speed
                bbox = np.transpose(curr_ped_seq[:, 2:-3])
                speed_curr_ped_seq = np.zeros(bbox.shape)
                # (seq[0] - seq[1]) -> speed 
                speed_curr_ped_seq[:, 1:] = bbox[:, 1:] - bbox[:, :-1]
                _idx = num_peds_considered
                # img dir
                curr_seq_dir[_idx, pad_front:pad_end, :] = curr_dir_seq.reshape(30, 1)
                # bbox
                curr_seq_bbox[_idx, pad_front:pad_end, :] = curr_ped_seq[:, 1:-1]
                # intention
                curr_seq_int[_idx, pad_front:pad_end, :] = curr_ped_seq[:, -1].reshape(30, 1)
                # speed
                curr_seq_speed[_idx, pad_front:pad_end, :] = np.transpose(speed_curr_ped_seq)
                # imu
                curr_seq_imu[_idx, pad_front:pad_end, :] = curr_imu_seq
                # neighbor
                curr_seq_neighbor[_idx, pad_front:pad_end, :] = curr_neighbor_seq
                num_peds_considered += 1

            if num_peds_considered >= 1:
                # one list[0] = cirrent data in 00006~00180
                seq_list_dir.append(curr_seq_dir[:num_peds_considered])
                seq_list_bbox.append(curr_seq_bbox[:num_peds_considered])
                seq_list_speed.append(curr_seq_speed[:num_peds_considered])
                seq_list_int.append(curr_seq_int[:num_peds_considered])
                seq_list_imu.append(curr_seq_imu[:num_peds_considered])
                seq_list_neigh.append(curr_seq_neighbor[:num_peds_considered])
        if len(seq_list_dir)>0:
            seq_dir = np.concatenate(seq_list_dir, axis=0)
            seq_bbox = np.concatenate(seq_list_bbox, axis=0)
            seq_speed = np.concatenate(seq_list_speed, axis=0)
            seq_int = np.concatenate(seq_list_int, axis=0)
            seq_imu = np.concatenate(seq_list_imu, axis=0)
            seq_neigh = np.concatenate(seq_list_neigh, axis=0)

        # make them together for save to npy
        if count == 0:
            seq_dir_arr = seq_dir
            seq_bbox_arr = seq_bbox
            seq_speed_arr = seq_speed
            seq_int_arr = seq_int
            seq_imu_arr = seq_imu
            seq_neigh_arr = seq_neigh
        else:
            seq_dir_arr = np.concatenate((seq_dir_arr, seq_dir), axis=0)
            seq_bbox_arr = np.concatenate((seq_bbox_arr, seq_bbox), axis=0)
            seq_speed_arr = np.concatenate((seq_speed_arr, seq_speed), axis=0)
            seq_int_arr = np.concatenate((seq_int_arr, seq_int), axis=0)
            seq_imu_arr = np.concatenate((seq_imu_arr, seq_imu), axis=0)
            seq_neigh_arr = np.concatenate((seq_neigh_arr, seq_neigh), axis=0)
    return seq_dir_arr.astype(str), seq_bbox_arr, seq_speed_arr, seq_int_arr, seq_imu_arr, seq_neigh_arr

def mask_num(n_data):
    p=np.zeros((len(n_data[:,0,0,0]),len(n_data[0,:,0,0]),1))
    for i in range(len(n_data[:,0,0,0])):
        ans = n_data[i,:,:,:]
        a=np.count_nonzero(ans,axis=2)
        tmp=np.count_nonzero(a,axis=1)
        p[i] = tmp.reshape(len(n_data[0,:,0,0]),1)  
    return p
