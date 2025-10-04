from matplotlib import path
import pandas as pd
from pathlib import Path
import numpy as np

procLoc = '../data/ppg-csv' # processed data location
procDir = Path(procLoc)
patientNums = np.arange(100001,100007,1)

# Function to add header to the processed data
# Load the original csv, add RGB header, and write back to the original csv file
def add_Header():
    for i in ['Left/', 'Right/']:
        for j in patientNums:
            path_to_file = procDir / i / (str(j) + '.csv')
            print(path_to_file)
            
            try:
                # Check if file already has headers
                with open(path_to_file, 'r') as f:
                    first_line = f.readline().strip()
                    if 'R,G,B' in first_line or 'R' in first_line:
                        print(f"File {path_to_file} already has headers, skipping.")
                        continue
                
                # Load data and add headers
                data_without_label = np.loadtxt(path_to_file, delimiter=',')
                df = pd.DataFrame(data_without_label, columns = ['R', 'G', 'B'])
                df.to_csv(path_to_file, index=False)
                print(f"Headers added to {path_to_file}")
                
            except Exception as e:
                print(f"Error processing {path_to_file}: {e}")

# Demo to load data from the csv file
if __name__ == "__main__":
    add_Header()
    # Test loading a file
    try:
        path_to_file = procDir / 'Left' / (str(patientNums[0]) + '.csv')
        data = np.genfromtxt(path_to_file, delimiter=',', skip_header=1).transpose()
        print(f"Successfully loaded data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading test data: {e}")