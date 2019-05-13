import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models
from dataset import TTADataset
from transforms import test_transform
from utils import mean_df, binarize_prediction, load
from utils import ThreadingDataLoader as DataLoader
from pprint import pprint

N_CLASSES = 1103
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_classes(item):
	return ' '.join(cls for cls, is_present in item.items() if is_present)

def main(config):
	"""
	model: trained model
	config: from config folder 
	"""
	model = getattr(models, config.model.name)(num_classes=N_CLASSES).to(device)
	load(model, f'./savings/{config.model.name}_fold{config.data.fold}/best_model.pt')

	sample = pd.read_csv('./dataset/sample_submission.csv')
	test_image_path = Path('./dataset/test/')
	test_dataset = TTADataset(test_image_path, sample, test_transform, 4)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.train.batch_size, num_workers=6)
	
	model.eval()
	all_outputs, all_ids = [], []
	with torch.no_grad():
		for images, image_ids in tqdm(test_loader):
			images = images.to(device)
			logits = model(images)
			probs = F.sigmoid(logits)
			all_outputs.append(probs.data.cpu().numpy())
			all_ids += image_ids

	all_outputs = np.concatenate(all_outputs)
	
	df = pd.DataFrame(
		data = all_outputs,
		index = all_ids,
		columns = list(map(str, range(N_CLASSES))))
	
	df = mean_df(df)

	sample = pd.read_csv('./dataset/sample_submission.csv', index_col='id')
	df = df.reindex(sample.index)
	df.to_csv(config.save_probs)
	
	out = binarize_prediction(df.values, config.threshold)
	df[:] = out
	df = df.apply(get_classes, axis=1)
	df.name = 'attribute_ids'
	
	submit_path = Path('./submissions/')
	if not submit_path.exists():
		submit_path.mkdir(parents=True, exist_ok=True)
	df.to_csv(submit_path/(f'{config.model.name}_{config.data.fold}_{config.threshold}.csv'), header=True)


def parse_args():
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--config', type=str)
	arg('--save_probs', type=str)
	arg('--threshold', type=float)
	args = parser.parse_args()

	with open(args.config) as f:
		config = yaml.load(f)

	config = {**config, **vars(args)} 

	return edict(config)


if __name__ == '__main__':
	config = parse_args()
	pprint(config)
	main(config)
