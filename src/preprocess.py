import os
import pandas as pd
import numpy as np
from scipy.signal import resample

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_dir = os.path.join(project_root, 'data', 'raw')
if not os.path.exists(os.path.join(raw_dir, 'Gaze')):
    raw_dir = os.path.join(project_root, 'data')

output_dir = os.path.join(project_root, 'data', 'processed')
os.makedirs(output_dir, exist_ok=True)

target_length = 500  # Fix méret (10 mp * 50 Hz)
window_size_sec = 10.0

pupil_cols = ['ET_PupilLeft', 'ET_PupilRight']
gaze_cols = ['ET_GazeLeftx', 'ET_GazeLefty']

X_dynamic = [] 
X_static = []  
y_targets = [] 
subject_ids = []

gaze_base_path = os.path.join(raw_dir, 'Gaze')
participants = sorted([d for d in os.listdir(gaze_base_path) if os.path.isdir(os.path.join(gaze_base_path, d))])
print(f"--- Fiziológiai Kalibráció (TEPR) Indítása {len(participants)} résztvevőn ---")

for participant_id in participants:
    labels_path = os.path.join(raw_dir, 'Labels', f'{participant_id}.csv')
    if not os.path.exists(labels_path):
        continue
        
    df_labels = pd.read_csv(labels_path)
    
    for session_idx in range(4):
        gaze_path = os.path.join(raw_dir, 'Gaze', participant_id, f'gaze_data_experiment_{session_idx}.csv')
        baseline_path = os.path.join(raw_dir, 'Gaze', participant_id, f'gaze_data_baseline_{session_idx}.csv')
        
        if os.path.exists(gaze_path) and os.path.exists(baseline_path):
            
            df_baseline = pd.read_csv(baseline_path)
            pupil_baseline_means = df_baseline[pupil_cols].mean().values
            
            df_gaze = pd.read_csv(gaze_path)
            session_labels = df_labels[f'level_{session_idx}'].dropna().values
            
            start_time = df_gaze['Timestamp'].iloc[0]
            df_gaze['Timestamp_norm'] = df_gaze['Timestamp'] - start_time
            
            for i, label in enumerate(session_labels):
                t_start = i * window_size_sec
                t_end = (i + 1) * window_size_sec
                
                mask = (df_gaze['Timestamp_norm'] >= t_start) & (df_gaze['Timestamp_norm'] < t_end)
                window_df = df_gaze[mask]
                
                if not window_df.empty:
                    pupil_data = window_df[pupil_cols].values
                    gaze_data = window_df[gaze_cols].values
                    
                    pupil_calibrated = pupil_data - pupil_baseline_means
                    
                    features = np.hstack([pupil_calibrated, gaze_data])
                    
                    features = np.nan_to_num(features, nan=0.0)
                    
                    features_resampled = resample(features, target_length, axis=0)
                    
                    X_dynamic.append(features_resampled)
                    X_static.append(session_idx)
                    y_targets.append(label)
                    subject_ids.append(participant_id)

print("\n--- FELDOLGOZÁS VÉGE ---")
X_dynamic = np.array(X_dynamic)
X_static = np.array(X_static)
y_targets = np.array(y_targets)
subject_ids = np.array(subject_ids)

print(f"Létrehozott Tenzor alakja: {X_dynamic.shape}")

pupil_mean = np.mean(X_dynamic[:, :, :2])
print(f"A kalibrált pupillaméretek átlaga: {pupil_mean:.4f} (Ennek kell 0-hoz közelinek lennie!)")

output_file = os.path.join(output_dir, 'processed_dataset_calibrated.npz')
np.savez(output_file, X_dynamic=X_dynamic, X_static=X_static, y=y_targets, subjects=subject_ids)
print(f"Adatok sikeresen mentve ide: {output_file}")