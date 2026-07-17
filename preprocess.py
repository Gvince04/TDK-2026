import os
from glob import glob
import pandas as pd

BASE_DIR = "./data"

gaze_dir   = os.path.join(BASE_DIR, "Gaze")
labels_dir = os.path.join(BASE_DIR, "Labels")

participants = [ d for d in os.listdir(gaze_dir) if os.path.isdir(os.path.join(gaze_dir, d)) ]
participants.sort()

print(f"Found {len(participants)} participant's directory.")

if len(participants) > 0:
    test_p_id = participants[0]
    print(f"Participant id: {test_p_id}")

    gaze_files = sorted(glob(os.path.join(gaze_dir, test_p_id, '*exp*.csv')))

    for gaze_file in gaze_files:
        filename = os.path.basename(gaze_file)
        
        label_file = os.path.join(labels_dir, test_p_id, filename.replace('gaze', 'label')) 
        
        print(f"\nLoading Session: {filename}")
        
        try:
            df_gaze = pd.read_csv(gaze_file)
            print(f" - Gaza data shape: {df_gaze.shape}")
            
            if os.path.exists(label_file):
                df_label = pd.read_csv(label_file)
                print(f" - Labels shape: {df_label.shape}")
            else:
                print(" - Warning: No label file found!")
                
        except Exception as e:
            print(f"Error reading the file: {e}")
            
else:
    print("No participants in the Gaze directory!")