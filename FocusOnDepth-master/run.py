import json
import torch
from glob import glob
from FOD.Predictor import Predictor

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('config.json', 'r') as f:
    config = json.load(f)

input_images = glob('input/*.jpg') + glob('input/*.png')

# Move the Predictor object to the GPU
predictor = Predictor(config, input_images, device=device)
predictor.run()
