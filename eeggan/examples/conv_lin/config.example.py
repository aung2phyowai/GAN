import sys
import os

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"

sys.path.append(r"C:\Users\hplis\OneDrive\Documents\BrainHackModels")

compiled_data_path = r'C:\Users\hplis\OneDrive\Documents\GitHub\train-768.pkl'
dataset_path = r'C:\Users\hplis\Downloads\eeg_files'
model_path = './test.cnt'