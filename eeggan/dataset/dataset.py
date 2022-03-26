import mne
from  scipy.signal import butter, buttord 
from  scipy.signal import iirnotch
from  scipy.signal import filtfilt
from scipy.signal import decimate
import numpy as np
from tqdm import tqdm

import os

multiplier = 1

class EEGDataClass():
    events = None

    def __init__(self, path):
        self.events = []

        for file in tqdm(os.listdir(path)):
            if '.set' in file:
                # Path should be def
                self.get_events(path + file)

    def get_events(self, fp):
        raw = mne.io.read_raw_eeglab(fp,eog='auto')
        Fs=raw.info['sfreq']
        Fs_int = int(Fs)

        [nb,na] = iirnotch(50,Q=30,fs=Fs)
        [b,a] = butter(2, 0.1, btype='highpass', fs=Fs)
        [b1,a1] = butter(2, 60, btype='lowpass', fs=Fs)

        syg=raw[:][0].copy()
        cz_index = raw.ch_names.index('Cz')
        sref = syg[cz_index].reshape(1, -1)-((syg[19]+syg[20])/2)

        sf = filtfilt(b,a,sref)
        sf = filtfilt(b1,a1,sf)
        sf = filtfilt(nb,na,sf)

        sf = decimate(sf, multiplier)
        Fs_new = int(Fs // multiplier)
        eventstarts = mne.events_from_annotations(
            raw.copy().resample(Fs_new)
        )[0]
        eventstarts=eventstarts[:,0][eventstarts[:,2]==1]

        ch_num = 1

        events = np.zeros((ch_num,len(eventstarts),int(Fs)))
        for i, ch in enumerate(sf[0:ch_num]):
            for j,st in enumerate(eventstarts):
                event = ch[int(-0.2*Fs_int+st):int(0.8*Fs_int+st)]
                init_mean = event[:int(0.2 * Fs_new)].mean()
                events[i,j] = event - init_mean

        self.events.append(events)

    def __getitem__(self, idx):
        return self.events[idx], True

    def __len__(self):
        return len(self.events)