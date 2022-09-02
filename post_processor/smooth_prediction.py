import numpy as np

def smooth_pred_window(preds, window_size=15):
    smoothed = np.array(preds).copy()
    for idx in range(window_size, len(preds)-window_size, window_size*2+1):
        tmp = smoothed[idx-window_size:(idx+window_size+1)]
        vals, counts = np.unique(tmp, return_counts=True)
        mode = np.argmax(counts)
        smoothed[idx-window_size:(idx+window_size+1)] = vals[mode]
    return smoothed
