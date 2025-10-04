# preprocess_data_spo2.py
# Library to create .h5 tensors from spo2 data in csv format

# Imports --------------------------------------------

import numpy as np
import h5py
import pandas as pd
from pathlib import Path

# Presets --------------------------------------------

PATIENT_NUMS = ['100001', '100002', '100003', '100004', '100005', '100006']

PROCESSED_DATA_DIR = '../data/ppg-csv/'
FEATS = ["left-r","left-g","left-b","right-r","right-g","right-b"]

GROUNDTRUTH_DATA = '../data/gt/'
GT_ROWS = ['SpO2 1', 'SpO2 2', 'SpO2 3', 'SpO2 4', 'SpO2 5']

# Functions --------------------------------------------

"""
Load patient data functions
"""
# Assumes 0,1,2 feats are for left hand and 3,4,5 feats are for right hand
def load_data_for_patient(pnum):
    # Read data from Left
    fpath_left = PROCESSED_DATA_DIR + 'Left/' + pnum + '.csv'
    try:
        data_left = np.genfromtxt(fpath_left, delimiter=',', skip_header=1).transpose()
    except Exception as e:
        print(f"Error loading left data for patient {pnum}: {e}")
        return None

    # read data from Right
    fpath_right = PROCESSED_DATA_DIR + 'Right/' + pnum + '.csv'
    try:
        data_right = np.genfromtxt(fpath_right, delimiter=',', skip_header=1).transpose()
    except Exception as e:
        print(f"Error loading right data for patient {pnum}: {e}")
        return None

    # Stack uneven arrays into single array
    max_len = max(data_left.shape[1], data_right.shape[1])
    data = np.zeros((6, max_len))
    data[:3, :data_left.shape[1]] = data_left
    data[3:, :data_right.shape[1]] = data_right

    return data

def load_data():
    # Load processed input data
    all_patient_data = []
    for pnum in PATIENT_NUMS:
        pdata = load_data_for_patient(pnum)
        if pdata is not None:
            all_patient_data.append(pdata)
        else:
            print(f"Skipping patient {pnum} due to loading error")
    
    if not all_patient_data:
        raise ValueError("No patient data could be loaded")
    
    max_len = max([pdata.shape[1] for pdata in all_patient_data])
    dataset = np.zeros((len(all_patient_data), len(FEATS), max_len))
    
    for i, pdata in enumerate(all_patient_data):
        dataset[i, :, :pdata.shape[1]] = pdata

    return dataset


"""
Load groundtruth data functions
"""
def load_groundtruth_for_patient(pnum):
    fpath = GROUNDTRUTH_DATA + pnum + '.csv'
    try:
        gt_data = pd.read_csv(fpath)
        
        # Extract relevant SpO2 columns
        spo2_data = []
        for col in GT_ROWS:
            if col in gt_data.columns:
                spo2_data.append(gt_data[col].values)
            else:
                print(f"Warning: Column '{col}' not found in {fpath}")
                # Use zeros as placeholder
                spo2_data.append(np.zeros(len(gt_data)))
        
        return np.array(spo2_data)
        
    except Exception as e:
        print(f"Error loading ground truth for patient {pnum}: {e}")
        return None


def load_groundtruth():
    all_patient_gt = []
    for pnum in PATIENT_NUMS:
        gt_data = load_groundtruth_for_patient(pnum)
        if gt_data is not None:
            all_patient_gt.append(gt_data)
        else:
            print(f"Skipping ground truth for patient {pnum}")
    
    if not all_patient_gt:
        raise ValueError("No ground truth data could be loaded")
    
    max_len = max([gt.shape[1] for gt in all_patient_gt])
    groundtruth = np.zeros((len(all_patient_gt), len(GT_ROWS), max_len))
    
    for i, gt in enumerate(all_patient_gt):
        groundtruth[i, :, :gt.shape[1]] = gt
    
    return groundtruth

def build_data_and_groundtruth():
    FPS = 30
    dataset = load_data()
    groundtruth = load_groundtruth()
    
    # Ensure time alignment
    clip_to = min(groundtruth.shape[2]*FPS, dataset.shape[2])
    groundtruth = groundtruth[:, :, :clip_to//30]
    dataset = dataset[:, :, :clip_to]

    # Create output directory if it doesn't exist
    output_dir = Path('../data/preprocessed/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'all_uw_data.h5'
    
    with h5py.File(output_path, 'w') as f:
        dset = f.create_dataset("dataset", tuple(dataset.shape), dtype='f')
        dset[:] = dataset
        dset.attrs['features_key'] = FEATS

        gt = f.create_dataset("groundtruth", tuple(groundtruth.shape), dtype='f')
        gt[:] = groundtruth
        gt.attrs['gt_keys'] = GT_ROWS
        
    print(f"Data saved to {output_path}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Groundtruth shape: {groundtruth.shape}")

def load_data_and_groundtruth():
    build_data_and_groundtruth()
    with h5py.File('../data/preprocessed/all_uw_data.h5', 'r') as f:
        data = f['dataset'][:]
        gt = f['groundtruth'][:]
    return data, gt

# Code to run --------------------------------------------

if __name__ == '__main__':
    data, gt = load_data_and_groundtruth()