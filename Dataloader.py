# Import libraries
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

from scipy import signal
from matplotlib import cm
import torch.utils.data as utils
import torch
import torch.nn as nn

training_data_dir = './data/lunar/training/data/S12_GradeA/'
event_list_file = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'

list_of_events = pd.read_csv(event_list_file)

def prepare_event_data_dict(list_of_events):
    list_of_event_ids = []
    event_data_dict = {}
    for event_idx in range(len(list_of_events)):
        event_data = list_of_events.iloc[event_idx]
        event_filename = event_data['filename']
        event_time_abs = event_data['time_abs(%Y-%m-%dT%H:%M:%S.%f)']
        event_time_rel = event_data['time_rel(sec)']
        event_id = event_data['evid']
        event_type = event_data['mq_type']
        list_of_event_ids.append(event_id)
        event_data_dict[event_id] = {'filename': event_filename, 'time_abs': event_time_abs, 'time_rel': event_time_rel, 'type': event_type}
    return list_of_event_ids, event_data_dict

list_of_event_ids, event_data_dict = prepare_event_data_dict(list_of_events)
list_of_event_times_datetimes = [datetime.strptime(event_data_dict[event_id]['time_abs'], '%Y-%m-%dT%H:%M:%S.%f') for event_id in list_of_event_ids]

def check_if_any_event_in_range(start_time, end_time):
    for e in list_of_event_times_datetimes:
        if e >= start_time and e <= end_time:
            return True
    return False

def prepare_data_loader(overlap, window_length, decimation_factor, spect_nfft, spect_nperseg, batch_size):
    all_spectrograms = []
    all_labels = []

    list_of_files = os.listdir(training_data_dir)
    list_of_files = [file for file in list_of_files if file.endswith('.mseed')]
    tr_data = None
    for file_idx in range(len(list_of_files)):
        current_event_date = list_of_files[file_idx].split('.')[4][:10]
        current_event_date = datetime.strptime(current_event_date, "%Y-%m-%d")
        if file_idx == len(list_of_files) - 1:
            days_difference = -1
        else:
            # print(current_event_date, end=' ')
            next_event_date = list_of_files[file_idx+1].split('.')[4][:10]
            next_event_date = datetime.strptime(next_event_date, "%Y-%m-%d")

            delta = next_event_date - current_event_date
            days_difference = delta.days

        mseed_file_path = f'{training_data_dir}{list_of_files[file_idx]}'
        st = read(mseed_file_path)

        start_time = st[0].stats.starttime.datetime
        sampling_rate = st[0].stats.sampling_rate

        tr = st.traces[0].copy()

        if tr_data is None:
            tr_data = tr.data
        else:
            tr_data = np.concatenate((tr_data, tr.data))
        # print(tr_data.shape)


        if days_difference == 1:
            continue
        else:
            iterator = 0
            samples_per_window = window_length*3600*sampling_rate
            list_of_spectrograms = []
            list_of_event_labels = []
            end_of_file = False
            while not end_of_file:
                #start_time + window_length hours

                tmp_data = tr_data[iterator:iterator+int(samples_per_window)]
                if len(tmp_data) < int(samples_per_window):
                    tmp_data = tr_data[-int(samples_per_window):]
                    end_of_file = True

                tmp_data_undersample = signal.decimate(tmp_data, decimation_factor, axis=0, zero_phase=True)
                _, _, sxx = signal.spectrogram(tmp_data_undersample, sampling_rate/decimation_factor, nfft=spect_nfft, nperseg=spect_nperseg)

                list_of_event_labels.append(check_if_any_event_in_range(start_time, start_time + timedelta(hours=window_length)))
                list_of_spectrograms.append(sxx)

                iterator += int(overlap*3600*sampling_rate)
                start_time = start_time + timedelta(hours=overlap)

            tr_data = None
            spectrograms_np_arr = np.array(list_of_spectrograms)
            labels_np_arr = np.array(list_of_event_labels)
            # print(spectrograms_np_arr.shape)

            all_spectrograms.append(spectrograms_np_arr)
            all_labels.append(labels_np_arr)
    all_spectrograms = np.concatenate(all_spectrograms, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    train_dataset = utils.TensorDataset(torch.from_numpy(all_spectrograms).float(),torch.from_numpy(all_labels).bool())
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    return train_loader