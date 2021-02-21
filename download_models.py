'''
Use this to download the models recommended by Andy Sandberg himself, probably trained by him
'''

from src.download_and_extract import download_and_extract_file, model_dict
import os

MODEL_DIR = './models'

os.makedirs(MODEL_DIR, exist_ok=True)
    
for model in model_dict.keys():
    download_and_extract_file(model, MODEL_DIR)