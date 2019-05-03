import os
import yaml
import tqdm
import shutil
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from datetime import datetime
from easydict import EasyDict as edict

import torch
import torch.optim as optim
import torch.nn as nn

import models
from transforms import train_transform, test_transform
from dataset import TrainDataset
from utils import ThreadingDataLoader as DataLoader, write_event, load, get_score, binarize_prediction


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES = 1103

def make_loader(df, image_transform, config):
	dataset = TrainDataset(Path(config.data.train_dir), df, image_transform, debug = False)
	dataloader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size, num_workers=6)
	return dataloader

def compute_my_loss(logits, target):
	criterion = nn.BCEWithLogitsLoss()
	loss = criterion(logits, target)
	return loss

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(model, train_loader, valid_loader, params, config, fresh=False):

	save = lambda epoch: torch.save({
		'model':model.state_dict(),
		'epoch': epoch,
		'train_losses':train_losses,
		'valid_losses':valid_losses,
		'lr':config.optimizer.params.lr,
		'best_valid_loss':best_valid_loss,
	}, f'./savings/{config.model.name}_fold{config.data.fold}/model.pt')

	init_optimizer = lambda params, lr: optim.Adam(params, lr)

	model_path = Path('./savings/')/(config.model.name + f'_fold{config.data.fold}')
	
	if not model_path.exists():
		model_path.mkdir(parents=True, exist_ok=True)
	
	if (model_path/'model.pt').exists():
		print('Loading saved model')
		state = load(model, str(model_path/'model.pt'))
		train_losses = state['train_losses']
		valid_losses = state['valid_losses']
		if len(valid_losses) > 0:
			valid_loss = valid_losses[-1]
		else:
			valid_loss = float('inf')
		best_valid_loss = state['best_valid_loss']
	else:
		train_losses = []
		valid_losses = []
		best_valid_loss = float('inf')
		valid_loss = float('inf')
	
	lr_changes = 0
	max_lr_changes = 4
	non_changed_epochs = 0
	
	optimizer = init_optimizer(params, lr=config.optimizer.params.lr)
	log_dir = Path(f'./savings/{config.model.name}_fold{config.data.fold}/train.log').open('at', encoding='utf8')
	
	
	try:
		for epoch in range(config.train.num_epochs):
			
			tq = tqdm.tqdm(total=len(train_loader) * config.batch_size)
			model.train()
			for step, (images, labels) in enumerate(train_loader):
				tq.set_description(f'Epoch {epoch}, Iteration {step}')

				images = images.to(device)
				labels = labels.to(device)

				logits = model(images)
				loss = compute_my_loss(logits, labels)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				train_losses.append(loss.item())

				tq.update(config.batch_size)
				mean_loss = np.mean(train_losses[-config.train.report_each:])
				tq.set_postfix(train_loss=f'{mean_loss:.5f}', valid_loss=valid_loss, best_valid_loss=best_valid_loss)
				
				if step and step % config.train.report_each == 0:
					write_event(log_dir, step, loss = f'{mean_loss:.5f}', valid_loss=f'{valid_loss:.5f}', best_loss=f'{best_valid_loss:.5f}')

			write_event(log_dir, step, loss = f'{mean_loss:.5f}', valid_loss=f'{valid_loss:.5f}', best_loss=f'{best_valid_loss:.5f}')				
			tq.close()
			save(epoch)

			metrics = validate(model, valid_loader)
			write_event(log_dir, step, **metrics)
			valid_loss = metrics['valid_loss']
			valid_losses.append(valid_loss)
		
			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
				shutil.copy(f'./savings/{config.model.name}_fold{config.data.fold}/model.pt', f'./savings/{config.model.name}_fold{config.data.fold}/best_model.pt')
				non_changed_epochs = 0
			else:
				non_changed_epochs += 1
				if non_changed_epochs > config.train.patience:
					lr_changes += 1
					if lr_changes > max_lr_changes:
						break
					lr = lr / 5
					non_changed_epochs = 0
					optimizer = init_optimizer(params, lr)
			if fresh:
				break
	
	except KeyboardInterrupt:
		tq.close()
		print(f'Trained Epochs = {epoch}, Iterations = {step}')
		return

def validate(model, valid_loader):
	model.eval()
	losses, targets, predictions = [], [], []
	
	with torch.no_grad():
		for valid_images, valid_labels in valid_loader:
			valid_images = valid_images.to(device)
			valid_labels = valid_labels.to(device)

			valid_logits = model(valid_images)
			loss = compute_my_loss(valid_logits, valid_labels)
			
			losses.append(loss.item())
			targets.append(valid_labels.cpu().numpy())
			predictions.append(torch.sigmoid(valid_logits).cpu().numpy())
	
	predictions = np.concatenate(predictions)
	targets = np.concatenate(targets)
	
	argsorted = predictions.argsort(axis=1)
	metrics = {}
	
	for threshold in np.linspace(0.1, 1, 10):
		metrics[f'valid_2f_th_{threshold:.2f}'] = get_score(targets, binarize_prediction(predictions, threshold, argsorted))
	
	metrics['valid_loss'] = np.mean(losses)
	
	return metrics

def main(config):
	folds = pd.read_csv(config.data.folds_dir)

	train_folds = folds[folds['fold'] != config.data.fold]
	valid_folds = folds[folds['fold'] == config.data.fold]
	
	train_loader = make_loader(train_folds, train_transform, config)
	valid_loader = make_loader(valid_folds, test_transform, config)
	
	
	model = getattr(models, config.model.name)(pretrained=True, num_classes=N_CLASSES).to(device)
	
	fresh_params = list(model.fresh_params())
	all_params = list(model.parameters())

	if config.mode == 'train':
		#train(model, train_loader, valid_loader, fresh_params, config, fresh=True)
		train(model, train_loader, valid_loader, all_params, config)
	elif config.mode == 'validate':
		metrics = validate(model, valid_loader)
		pprint(metrics)

def parse_args():
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--config', type=str)
	arg('--batch_size', type=int, default=32)
	args = parser.parse_args()

	with open(args.config) as f:
		config = yaml.load(f)
	

	return edict(config)

def get_args(config_path):
	

	return config

if __name__ =='__main__':
	seed_torch()
	config = parse_args()
	
	pprint(config)
	main(config)

# def train_one_batch():


# 	images = np.load('./data/one_batch_images.npy')
# 	labels = np.load('./data/one_batch_labels.npy')
	

# 	model = getattr(models, 'resnet50')(pretrained=True, num_classes=N_CLASSES).to(device)
	
# 	train_losses = []
	
# 	model.train()
# 	images = torch.from_numpy(images).to(device)
# 	labels = torch.from_numpy(labels).to(device)
	
# 	images = images[0].unsqueeze(0)
# 	labels = labels[0].unsqueeze(0)
# 	print(labels.cpu().numpy().sum())

# 	optimizer = optim.Adam(list(model.fresh_params()), 0.1)
# 	try:
# 		for epoch in range(1000):
			

# 			logits = model(images)
# 			loss = compute_my_loss(logits, labels)

# 			optimizer.zero_grad()
# 			loss.backward()
# 			optimizer.step()

# 			train_losses.append(loss.item())

# 			print('Loss = ', train_losses[-1])
			
# 	except KeyboardInterrupt:
# 		return

# 	optimizer = optim.Adam(list(model.params()), 0.0001)
# 	try:
# 		for epoch in range(1000):
			
# 			logits = model(images)
# 			loss = compute_my_loss(logits, labels)

# 			optimizer.zero_grad()
# 			loss.backward()
# 			optimizer.step()

# 			train_losses.append(loss.item())

# 			mean_loss = np.mean(train_losses[-10:])
# 			print(mean_loss)
			
# 	except KeyboardInterrupt:
# 		return

# if __name__ == '__main__':
# 	seed_torch()
# 	train_one_batch()
