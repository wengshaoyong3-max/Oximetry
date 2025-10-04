# Oximetry-phone-cam-data

## Purpose
**Open source data for smartphone camera oximetry, sensing SpO2 and hypoxemia risk on a clinically relevant spread of data**

This repository contains the open source data from the smartphone camera oximetry study by [Hoffman et al in 2022](https://www.nature.com/articles/s41746-022-00665-y).  It can be used to attempt to infer blood oxygen saturation (SpO2) and classify risk of hypoxemia using videos gathered via a smartphone camera using machine learning or analytical methods.  The data is the first gathered using a smartphone camera on a clinically relevant spread of SpO2 levels (70%-100%).

The data was gathered by researchers at the University of Washington and the University of California, San Diego, and is provided free and open source for the community to use for future projects.  More information can be found in the publication

## Getting Started
Clone the repo and run the hypoxemia prediction software to get started!

## Data Processing Pipeline

If you want to process raw video data from scratch, follow these steps:

### Step 1: Prepare Raw Video Data
Place videos in the `../data/raw-videos/raw` folder:
- Left hand videos in `/Left` folder
- Right hand videos in `/Right` folder

### Step 2: Process Videos to Extract RGB Values
Run `process_data_spo2.py`. This script uses OpenCV to process MP4 videos (calculating average RGB values per frame):
```bash
python process_data_spo2.py
```
**Note:** OpenCV uses BGR protocol for image processing, which may differ from other tools.

### Step 3: Add Headers to CSV Files
Run the `add_Header()` function from `header_data_spo2.py`. This function adds RGB headers to the CSV files processed by the process_data_spo2 script:
```bash
python -c "from header_data_spo2 import add_Header; add_Header()"
```

### Step 4: Create H5 Dataset
Run `preprocess_data_spo2.py` to load all preprocessed RGB values and ground truth data and package them into a single .h5 file:
```bash
python preprocess_data_spo2.py
```

### Step 5: Run Hypoxemia Prediction
Finally, run the main prediction software:
```bash
python hypoxemia_predictor.py
```

### Quick Setup

1. **Install Required Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install OpenCV for Video Processing (if needed):**
```bash
pip install opencv-python
```

3. **Run the Hypoxemia Prediction Software:**
```bash
python hypoxemia_predictor.py
```

More example code can be found in the examples directory using the preprocessed data.  If you want to use the raw video data, please see the "Data Processing Pipeline" section below.

### Needed packages: 
* numpy
* pandas  
* scikit-learn
* matplotlib
* h5py
* gradio
* opencv-python (for video processing)

## Data Format
There were 6 subjects in this study (numbered 100001-100006).

The smartphone oximetry data was collected in the form of MP4 videos, located in `../data/raw-videos/raw/`. Each frame's R, G, and B values are averaged to create the csv files in `../data/ppg-csv/`.

The ground truth data was collected from four standard pulse oximeters attached to the subjects' other fingers. That data can be found in `../data/gt/`.

### Project Structure
```
../data/
├── raw-videos/raw/           # Raw MP4 video files
│   ├── Left/                 # Left hand videos
│   └── Right/                # Right hand videos  
├── ppg-csv/                  # Processed RGB CSV files
│   ├── Left/                 # Left hand processed data
│   └── Right/                # Right hand processed data
├── gt/                       # Ground truth SpO2 data
├── info/                     # Video metadata files
└── preprocessed/             # Final processed data
    ├── all_uw_data.h5       # Training dataset
    └── linreg_preds.csv     # Prediction results
```

### Data Format Notes
* Camera framerate = 30 Hz
* Ground truth pulse oximeters framerate = 1 Hz  
* Recording was started and stopped on the camera and the pulse oximeters at the same time

## Background
We performed a Varied Fractional Inspired Oxygen (Varied FiO2) study, which is a clinical development validation study in which test subjects are administered a controlled mixture of oxygen and nitrogen to lower their SpO2 level over a period of 12-16 minutes.  The patients had one finger from each hand on a phone camera, while the camera flash transmitted light through their fingertips for reflectance photoplethysmography at the Red, Green, and Blue wavelengths.

### Ideas
Go ahead and try different models:
* Analytical (eg. ratio-of-ratios)
* Deep Learning
* Linear Regression
* Or, think of your own!

### Ground Truth Labels
A metadata file can be found in data/gt/metadata.csv, which describes the fields listed in the metadata files.  A table is also included below:
| Label        | Description                                                           |
|--------------|-----------------------------------------------------------------------|
| SpO2 1       | SpO2 reading from PPG of pulse ox 1 (3900P TT+ 9.000/11.000) (%)      |
| SpO2 2       | SpO2 reading from PPG of pulse ox 2 (Nellcor N-600X V 1.6.0.0) (%)    |
| SpO2 3       | Unfilled signal from pulse ox 3 (Safety Oxim 3 ECG Datex-Ohmeda S5)   |
| SpO2 4       | SpO2 reading from PPG of pulse ox 4 (Nellcor N-600X V 1.6.0.0) (%)    |
| SpO2 5       | SpO2 reading from PPG of pulse ox 5 (Masimo Radical 7 Rainbow II) (%) |
| Pulse 1      | Heart rate from PPG of pulse ox 1 (3900P TT+ 9.000/11.000) (bpm)      |
| Pulse 2      | Heart rate from PPG of pulse ox 2 (Nellcor N-600X V 1.6.0.0) (bpm)    |
| Pulse 3      | Unfilled signal from pulse ox 3 (Safety Oxim 3 ECG Datex-Ohmeda S5)   |
| Pulse 4      | Heart rate from PPG of pulse ox 4 (Nellcor N-600X V 1.6.0.0) (bpm)    |
| Pulse 5      | Heart rate from PPG of pulse ox 5 (Masimo Radical 7 Rainbow II) (bpm) |
| PI 1         | Perfusion Index from PPG of pulse ox 1 (3900P TT+ 9.000/11.000)       |
| PI 2         | Perfusion Index from PPG of pulse ox 2 (Nellcor N-600X V 1.6.0.0)     |
| PI 3         | Unfilled signal from pulse ox 3 (Safety Oxim 3 ECG Datex-Ohmeda S5)   |
| PI 4         | Perfusion Index from PPG of pulse ox 4 (Nellcor N-600X V 1.6.0.0)     |
| PI 5         | Perfusion Index from PPG of pulse ox 5 (Masimo Radical 7 Rainbow II)  |
| ECG 3        | Heart rate from ECG of pulse ox 3 (Safety Oxim 3 ECG Datex-Ohmeda S5) |
| Rig FiO2     | Percentage of oxygen delivered to subjec in gas mixture (%)           |

## Citation
If you use this data or code in your project, please cite it.  Here's the APA format:

Hoffman, J. S., Viswanath, V. K., Tian, C., Ding, X., Thompson, M. J., Larson, E. C., Patel, S. N., & Wang, E. J. (2022). Smartphone camera oximetry in an induced hypoxemia study. _NPJ digital medicine_, 5(1), 146.

### License
This data is provided open-source via the MIT license.  For more details, see the [LICENSE file](https://github.com/ubicomplab/oximetry-phone-cam-data/blob/dev3/LICENSE).  We want you to use it for whatever creative projects you can come up with!  

