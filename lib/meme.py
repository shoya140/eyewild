# encode utf-8

import numpy as np
import pandas as pd
from scipy import signal

def detect_blink(timestamps, vv, nf=200, cutoff=100,
                 th_right=0.5, th_up_to_down=1.0,
                 window_size=10):
    b, a = signal.butter(2, 1 - (nf - cutoff)/nf)
    y = signal.filtfilt(b, a, vv)
    n_y = [(v - np.mean(y))/np.std(y) for v in y]

    t_rep = 0
    d_rep = []
    output = {
        'timestamp': [],
        'blink': [],
    }
    for t, d in zip(timestamps, n_y):
        if t_rep == 0:
            t_rep = t
        d_rep.append(d)
        if t - t_rep >= window_size:
            output['timestamp'].append((t-t_rep)/2 + t_rep)
            pos_max = [i for i, j in enumerate(d_rep) if j == np.max(d_rep)][0]
            pos_min = [i for i, j in enumerate(d_rep) if j == np.min(d_rep)][0]

            if np.max(d_rep) > th_right and np.max(d_rep) - np.min(d_rep) > th_up_to_down and pos_max < pos_min:
                output['blink'].append(1)
            else:
                output['blink'].append(0)
            t_rep = 0
            d_rep = []

    return pd.DataFrame(index=output['timestamp'], data={
        'blink': output['blink']
    })