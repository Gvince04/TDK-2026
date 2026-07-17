import os
import pandas as pd

BASE_DIR = "./data"

participant_id = '1026'

print(f"--- Processing {participant_id}. participant's data ---")

labels_path = os.path.join(BASE_DIR, 'Labels', f'{participant_id}.csv')
df_labels = pd.read_csv(labels_path)

for session_idx in range(4):
    print(f"\n[ Session: {session_idx} ]")

    gaze_filename = f'gaze_data_experiment_{session_idx}.csv'
    gaze_path = os.path.join(BASE_DIR, 'Gaze', participant_id, gaze_filename)

    if os.path.exists(gaze_path):
        df_gaze = pd.read_csv(gaze_path)
        
        label_column = f'level_{session_idx}'
        session_labels = df_labels[label_column].dropna() 
        
        num_labels = len(session_labels)
        num_gaze_rows = len(df_gaze)
        expected_gaze_rows = num_labels * 500 
        
        print(f" - Loaded Gaze row count: {num_gaze_rows}")
        print(f" - Loaded Labels count: {num_labels} (in {num_labels * 10} seconds)")
        print(f" - Expected gaze row count: {expected_gaze_rows}")
        
        if session_idx == 0:
            print(f" - A szemkövetés oszlopai: {list(df_gaze.columns)[:5]} ...")
    else:
        print(f" - Hiba: Nem található a fájl: {gaze_path}")