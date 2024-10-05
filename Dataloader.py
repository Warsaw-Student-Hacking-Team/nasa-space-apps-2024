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

from torch.utils.data import WeightedRandomSampler

# def average_in_second_axis(data, factor):
#     if data.shape[1] % factor == 0:
#         return np.mean(data.reshape(data.shape[0], -1, factor), axis=2)
#     else:
#         num_full_groups = data.shape[1] // factor
#         full_groups = np.mean(data[:, :num_full_groups * factor].reshape(data.shape[0], -1, factor), axis=2)
#         remainder_group = np.mean(data[:, num_full_groups * factor:], axis=1, keepdims=True)
#         return np.concatenate((full_groups, remainder_group), axis=1)

# def apply_filter(st, minfreq, maxfreq):
#     st_filt = st.copy()
#     st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
#     tr_filt = st_filt.traces[0].copy()
#     tr_data_filt = tr_filt.data

#     f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)
#     return f, t, sxx

# def spectrogram_plot(tr_times_filt, tr_data_filt, t, f, sxx, cm):
#     fig = plt.figure(figsize=(6, 6))
#     ax2 = plt.subplot(1, 1, 1)
#     vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
#     # ax2.set_xlim([min(tr_times_filt),max(tr_times_filt)])
#     ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
#     ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
#     cbar = plt.colorbar(vals, orientation='horizontal')
#     cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

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

def check_if_any_event_in_range(list_of_event_times_datetimes,start_time, end_time):
    for e in list_of_event_times_datetimes:
        if e >= start_time and e <= end_time:
            return True
    return False

def get_uniqe_dates(list_of_event_times):
    unique_dates = []
    unique_files = []
    for e in list_of_event_times:
        if e.split('.')[4][:10] not in unique_dates:
            unique_dates.append(e.split('.')[4][:10])
            unique_files.append(e)
    return unique_files

def prepare_data_loader(overlap, window_length, decimation_factor, spect_nfft, spect_nperseg, batch_size, data_dir, labels_file_path=None):

    if labels_file_path is not None:
        list_of_events = pd.read_csv(labels_file_path)
        list_of_event_ids, event_data_dict = prepare_event_data_dict(list_of_events)
        list_of_event_times_datetimes = [datetime.strptime(event_data_dict[event_id]['time_abs'], '%Y-%m-%dT%H:%M:%S.%f') for event_id in list_of_event_ids]
        all_labels = []

    all_spectrograms = []

    list_of_files = os.listdir(data_dir)
    list_of_files = [file for file in list_of_files if file.endswith('.mseed')]
    # print(len(list_of_files))
    print(f'Starting number of files:{len(list_of_files)}')
    list_of_files = get_uniqe_dates(list_of_files)
    print(f'Number of uniques files:{len(list_of_files)}')

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

        mseed_file_path = f'{data_dir}{list_of_files[file_idx]}'
        st = read(mseed_file_path)

        tr = st.traces[0].copy()

        if tr_data is None:
            start_time = st[0].stats.starttime.datetime
            sampling_rate = st[0].stats.sampling_rate
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
            if labels_file_path is not None:
                list_of_event_labels = []
            end_of_file = False
            while not end_of_file:
                #start_time + window_length hours

                tmp_data = tr_data[iterator:iterator+int(samples_per_window)]
                end_time = start_time + timedelta(hours=window_length)
                if len(tmp_data) < int(samples_per_window):
                    tmp_data = tr_data[-int(samples_per_window):]
                    end_of_file = True

                if end_time > st[0].stats.endtime.datetime:
                    end_time = st[0].stats.endtime.datetime

                tmp_data_undersample = signal.decimate(tmp_data, decimation_factor, axis=0, zero_phase=True)
                _, _, sxx = signal.spectrogram(tmp_data_undersample, sampling_rate/decimation_factor, nfft=spect_nfft, nperseg=spect_nperseg)
                list_of_spectrograms.append(sxx)

                if labels_file_path is not None:
                    list_of_event_labels.append(check_if_any_event_in_range(list_of_event_times_datetimes, start_time, end_time))
                # if (check_if_any_event_in_range(list_of_event_times_datetimes, start_time, end_time)):
                #     t, f, sxx = signal.spectrogram(tmp_data_undersample, sampling_rate/decimation_factor, nfft=spect_nfft, nperseg=spect_nperseg)
                #     print(start_time, end_time)
                    
                iterator += int(overlap*3600*sampling_rate)
                start_time = start_time + timedelta(hours=overlap)

            tr_data = None
            spectrograms_np_arr = np.array(list_of_spectrograms)
            all_spectrograms.append(spectrograms_np_arr)

            if labels_file_path is not None:
                labels_np_arr = np.array(list_of_event_labels)
                all_labels.append(labels_np_arr)

    all_spectrograms = np.concatenate(all_spectrograms, axis=0)

    #normalize spectrograms
    print(np.min(all_spectrograms), np.max(all_spectrograms))
    all_spectrograms = (all_spectrograms - np.mean(all_spectrograms))/np.std(all_spectrograms)
    print(np.min(all_spectrograms), np.max(all_spectrograms))

    print(all_spectrograms.shape)
    if labels_file_path is not None:
        all_labels = np.concatenate(all_labels, axis=0)
        print(all_labels.shape)
        print(f'Number of windows with seismic event {np.sum(all_labels)}')
        train_dataset = utils.TensorDataset(torch.from_numpy(all_spectrograms).float(),torch.from_numpy(all_labels).bool())

        noise_sample_weight = np.sum(all_labels)/all_labels.shape[0]
        event_sample_weight = 1 - noise_sample_weight
        print(f'Noise sample weight: {noise_sample_weight}')
        print(f'Event sample weight: {event_sample_weight}')

        #prepare sample weights for sampler
        sample_weights = np.zeros(all_labels.shape[0])
        sample_weights[all_labels] = event_sample_weight
        sample_weights[~all_labels] = noise_sample_weight
        sample_weights = torch.from_numpy(sample_weights).float()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, sampler=sampler)

    else:
        train_dataset = utils.TensorDataset(torch.from_numpy(all_spectrograms).float())
        train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    return train_loader
