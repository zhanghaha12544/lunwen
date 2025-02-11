import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from FOD.Trainer import Trainer
from FOD.dataset import AutoFocusDataset

with open('config.json', 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(4)

list_data = config['Dataset']['paths']['list_datasets']

## train set
autofocus_datasets_train = []
for dataset_name in list_data:
    autofocus_datasets_train.append(AutoFocusDataset(config, dataset_name, 'train'))
train_data = ConcatDataset(autofocus_datasets_train)
# Move train data to GPU
train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

## validation set
autofocus_datasets_val = []
for dataset_name in list_data:
    autofocus_datasets_val.append(AutoFocusDataset(config, dataset_name, 'val'))
val_data = ConcatDataset(autofocus_datasets_val)
# Move validation data to GPU
val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

trainer = Trainer(config)
# Move trainer to GPU
trainer.to(device)
trainer.train(train_dataloader, val_dataloader)
