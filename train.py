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
from transforms import transform_func #train_transform, test_transform,
from dataset import TrainDataset
from utils import ThreadingDataLoader as DataLoader, write_event, load, get_score, binarize_prediction#, MyDataParallel
#from loss import FocalLoss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES = 1103

def make_loader(df, image_transform, config):
	dataset = TrainDataset(Path(config.data.train_dir), df, image_transform, debug = False)
	num_workers = 6
	if torch.cuda.device_count() > 1:
		num_workers = 48
	dataloader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size, num_workers=num_workers)
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
		'lr':lr,
		'best_valid_loss':best_valid_loss,
		'best_f2score':best_f2score,
	}, f'./savings/{config.model.name}_fold{config.fold}/model.pt')

	init_optimizer = lambda params, lr: optim.Adam(params, lr)

	model_path = Path('./savings/')/(config.model.name + f'_fold{config.fold}')
	
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
		best_f2score = state['best_f2score']
	else:
		train_losses = []
		valid_losses = []
		best_valid_loss = float('inf')
		best_f2score = 0
		valid_loss = float('inf')
	
	#lr_changes = 0
	#max_lr_changes = 4
	non_changed_epochs = 0
	lr = config.lr
	optimizer = init_optimizer(params, lr=lr)
	log_dir = Path(f'./savings/{config.model.name}_fold{config.fold}/train.log').open('at', encoding='utf8')
	
	try:
		for epoch in range(config.train.num_epochs):
			tq = tqdm.tqdm(total=len(train_loader) * config.batch_size)
			model.train()
			for step, (images, labels) in enumerate(train_loader):
				tq.set_description(f'Epoch {epoch}, Iteration {step}')

				images = images.to(device)
				labels = labels.to(device)

				logits = model(images)
				#logits = torch.sigmoid(logits)
				
				loss = compute_my_loss(logits, labels)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				train_losses.append(loss.item())

				tq.update(config.batch_size)
				mean_loss = np.mean(train_losses[-config.train.report_each:])
				tq.set_postfix(train_loss=f'{mean_loss:.5f}', 
					valid_loss=valid_loss, 
					best_valid_loss=best_valid_loss,
					best_f2score = best_f2score)
				
				if step and step % config.train.report_each == 0:
					write_event(log_dir, step, 
						loss = f'{mean_loss:.5f}', 
						valid_loss=f'{valid_loss:.5f}', 
						best_loss=f'{best_valid_loss:.5f}',
						best_f2score = f'{best_f2score:.5f}')


			write_event(log_dir, step, 
				loss = f'{mean_loss:.5f}', 
				valid_loss=f'{valid_loss:.5f}', 
				best_loss=f'{best_valid_loss:.5f}',
				best_f2score = f'{best_f2score:.5f}')

			tq.close()
			save(epoch)

			metrics = validate(model, valid_loader)
			write_event(log_dir, step, **metrics)
			valid_loss = metrics['valid_loss']
			valid_losses.append(valid_loss)

			current_f2score = metrics['max_f2score']

			if current_f2score > best_f2score:
				best_f2score = current_f2score
				shutil.copy(f'./savings/{config.model.name}_fold{config.fold}/model.pt', f'./savings/{config.model.name}_fold{config.fold}/best_model.pt')
				non_changed_epochs = 0
			else:
				non_changed_epochs += 1
				if non_changed_epochs > config.train.patience:
					lr = lr / 5
					non_changed_epochs = 0
					optimizer = init_optimizer(params, lr) 

			# if valid_loss < best_valid_loss:
			# 	best_valid_loss = valid_loss
			# 	shutil.copy(f'./savings/{config.model.name}_fold{config.fold}/model.pt', f'./savings/{config.model.name}_fold{config.fold}/best_model.pt')
			# 	non_changed_epochs = 0
			# else:
			# 	non_changed_epochs += 1
			# 	if non_changed_epochs > config.train.patience:
			# 		lr_changes += 1
			# 		if lr_changes > max_lr_changes:
			# 			break
			# 		lr = lr / 5
			# 		non_changed_epochs = 0
			# 		optimizer = init_optimizer(params, lr)
			
			if fresh:
				break
	
	except KeyboardInterrupt:
		tq.close()
		print(f'Trained Epochs = {epoch}, Iterations = {step}, lr = {lr}')
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
	np.save('./savings/predictions.npy', predictions)
	np.save('./savings/targets.npy', targets)
	
	argsorted = predictions.argsort(axis=1)
	metrics = {}
	
	for threshold in np.linspace(0.1, 1, 10):
		metrics[f'valid_2f_th_{threshold:.2f}'] = get_score(targets, binarize_prediction(predictions, threshold, argsorted))
	
	metrics['valid_loss'] = np.mean(losses)
	

	f2_scores = []
	for k, v in metrics.items():
		if 'valid_2f_th_' in k:
			f2_scores.append(v)

	metrics['max_f2score'] = max(f2_scores)

	return metrics

def main(config):
	folds = pd.read_csv(config.data.folds_dir)

	train_folds = folds[folds['fold'] != config.fold]
	valid_folds = folds[folds['fold'] == config.fold]
	train_transform = transform_func(config.model.input_shape)
	test_transform = transform_func(config.model.input_shape)
	
	train_loader = make_loader(train_folds, train_transform, config)
	valid_loader = make_loader(valid_folds, test_transform, config)
	
	model = getattr(models, config.model.name)(pretrained=True, num_classes=N_CLASSES)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model = model.to(device)
	if torch.cuda.device_count() > 1:
		fresh_params = list(model.module.fresh_params())
		all_params = list(model.module.parameters())
	else:
		fresh_params = list(model.fresh_params())
		all_params = list(model.parameters())

	
	if config.mode == 'train':
		train(model, train_loader, valid_loader, fresh_params, config, fresh=True)
		train(model, train_loader, valid_loader, all_params, config)
	elif config.mode == 'validate':
		model_path = Path('./savings/')/(config.model.name + f'_fold{config.fold}')
		_ = load(model, str(model_path/'best_model.pt'))
		metrics = validate(model, valid_loader)
		pprint(metrics)

def parse_args():
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--mode', type=str, default='train')
	arg('--config', type=str)
	arg('--batch_size', type=int, default=32)
	arg('--lr', type=float)
	arg('--fold', type=int)
	args = parser.parse_args()

	with open(args.config) as f:
		config = yaml.load(f)
	config = {**config, **vars(args)}

	return edict(config)

if __name__ =='__main__':
	seed_torch()
	config = parse_args()
	
	pprint(config)
	main(config)