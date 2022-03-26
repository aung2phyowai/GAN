import mne
from  scipy.signal import butter, buttord 
from  scipy.signal import iirnotch
from  scipy.signal import filtfilt
from scipy.signal import decimate


class EEGDataClass(Dataset):
    events = None

    def __init__(self, dir):
        self.events = []

        for file in os.listdir(dir):
            if '.set' in file:
                self.get_events(file)

    def get_events(self, fp):
        raw = mne.io.read_raw_eeglab(fp,eog='auto')
        Fs=raw.info['sfreq']

        [nb,na] = iirnotch(50,Q=30,fs=Fs)
        [b,a] = butter(2, 0.1, btype='highpass', fs=Fs)
        [b1,a1] = butter(2, 60, btype='lowpass', fs=Fs)

        sf = filtfilt(b,a,sref)
        sf = filtfilt(b1,a1,sf)
        sf = filtfilt(nb,na,sf)

        sf = decimate(sf, 8)
        Fs_new = int(Fs // 8)
        eventstarts = mne.events_from_annotations(
            raw.copy().resample(256)
        )[0]
        eventstarts=eventstarts[:,0][eventstarts[:,2]==1]

        ch_num = 1

        events = np.zeros((ch_num,len(eventstarts),int(Fs)))
        for i, ch in enumerate(sf[0:ch_num]):
            for j,st in enumerate(eventstarts):
                event = ch[int(-0.2*Fs_int+st):int(0.8*Fs_int+st)]
                init_mean = event[:int(0.2 * Fs_new)].mean()
                events[i,j] = event - init_mean

        self.events += events

    def __getitem__(self, idx):
        return self.events[idx], True

    def __len__(self):
        return len(self.events)