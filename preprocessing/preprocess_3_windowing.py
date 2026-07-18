import os
import pandas as pd
import numpy as np

base_dir = './data'
participant_id = '1026'

print(f"--- Processing {participant_id}. participant's data ---")

labels_path = os.path.join(base_dir, 'Labels', f'{participant_id}.csv')
df_labels = pd.read_csv(labels_path)

all_windows = []
all_static  = []
all_targets = []

for session_idx in range(4):
    gaze_path = os.path.join(base_dir, 'Gaze', participant_id, f'gaze_data_experiment_{session_idx}.csv')
    
    if os.path.exists(gaze_path):
        print(f"\n[ Session: {session_idx} ]")
        df_gaze = pd.read_csv(gaze_path)
        
        session_labels = df_labels[f'level_{session_idx}'].dropna().values
        
        start_time = df_gaze['Timestamp'].iloc[0]
        df_gaze['Timestamp_norm'] = df_gaze['Timestamp'] - start_time
        
        window_size_sec = 10.0
        
        for i, label in enumerate(session_labels):
            t_start = i * window_size_sec
            t_end = (i + 1) * window_size_sec
            
            mask = (df_gaze['Timestamp_norm'] >= t_start) & (df_gaze['Timestamp_norm'] < t_end)
            window_df = df_gaze[mask]
            
            if not window_df.empty:
                features = window_df[['ET_PupilLeft', 'ET_PupilRight', 'ET_GazeLeftx', 'ET_GazeLefty']].copy()
                
                all_windows.append(features)
                all_static.append(session_idx)
                all_targets.append(label)
            else:
                print(f" - Error: Empy window at index: {i}. ({t_start}s - {t_end}s)")

print("\n--- Preprocess End ---")
print(f"Created windows count: {len(all_windows)}")
print(f"Avarage window size (row count): {len(all_windows[0])}")
print(f"Example: First window's static context {all_static[0]}, target: {all_targets[0]}")