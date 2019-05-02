from pathlib import Path
import pandas as pd
import tqdm
import numpy as np
from datetime import datetime
import shutil
import json
from pprint import pprint
import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import random
import models
from transforms import train_transform, test_transform
from dataset import TrainDataset, TTADataset
from utils import ThreadingDataLoader as DataLoader

from sklearn.metrics import fbeta_score

from PIL import Image
from torchvision import transforms
from dataset import load_transform_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = Path('../data/')
train_root = DATA_ROOT/'train'
test_root = DATA_ROOT/'test'

fold_n = 4
batch_size = 32
N_CLASSES = 1103
patience = 10

def mean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).mean()


def make_loader(df, image_transform, args):
	dataset = TrainDataset(train_root, df, image_transform, debug = False)
	dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=6)
	return dataloader

def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = str(datetime.now())
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()
	
	
def load(model, path):
	state = torch.load(path)
	model.load_state_dict(state['model'])
	return state

init_optimizer = lambda params, lr: optim.Adam(params, lr)

def compute_my_loss(logits, target):
	criterion = nn.BCEWithLogitsLoss()
	loss = criterion(logits, target)
	return loss

def get_score(targets, y_pred):
	return fbeta_score(targets, y_pred, beta = 2, average='samples')

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

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(model, train_loader, valid_loader, params, args, fresh=False):

	save = lambda epoch: torch.save({
		'model':model.state_dict(),
		'epoch': epoch,
		'train_losses':train_losses,
		'valid_losses':valid_losses,
		'lr':lr,
		'best_valid_loss':best_valid_loss,
	}, f'./savings/{model.name}_fold{args.fold}/model.pt')


	model_path = Path('./savings/')/(model.name + f'_fold{args.fold}')
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
	epoch = 0
	lr = 1e-4
	
	lr_changes = 0
	max_lr_changes = 4
	non_changed_epochs = 0
	
	optimizer = init_optimizer(params, lr=lr)
	log_dir = Path(f'./savings/{model.name}_fold{args.fold}/train.log').open('at', encoding='utf8')
	report_each = 10
	num_epochs = 6
	try:
		for epoch in range(num_epochs):
			epoch += 1
			tq = tqdm.tqdm(total=len(train_loader) * batch_size)
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

				tq.update(batch_size)
				mean_loss = np.mean(train_losses[-report_each:])
				tq.set_postfix(train_loss=f'{mean_loss:.5f}', valid_loss=valid_loss, best_valid_loss=best_valid_loss)

				if step and step % report_each == 0:
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
				shutil.copy(f'./savings/{model.name}_fold{args.fold}/model.pt', f'./savings/{model.name}_fold{args.fold}/best_model.pt')
				non_changed_epochs = 0
			else:
				non_changed_epochs += 1
				if non_changed_epochs > patience:
					lr_changes += 1
					if lr_changes > max_lr_changes:
						break
					lr = lr/5
					non_changed_epochs = 0
					optimizer = init_optimizer(params, lr)

			if fresh:
				break

	except KeyboardInterrupt:
		tq.close()
		print(f'Trained Epochs = {epoch}, Iterations = {step}')

def binarize_prediction(probabilities, threshold: float, argsorted=None,
						min_labels=1, max_labels=10):
	assert probabilities.shape[1] == N_CLASSES
	if argsorted is None:
		argsorted = probabilities.argsort(axis=1)
	max_mask = _make_mask(argsorted, max_labels)
	min_mask = _make_mask(argsorted, min_labels)
	prob_mask = probabilities > threshold
	return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
	mask = np.zeros_like(argsorted, dtype=np.uint8)
	col_indices = argsorted[:, -top_n:].reshape(-1)
	row_indices = [i // top_n for i in range(len(col_indices))]
	mask[row_indices, col_indices] = 1
	return mask

def get_classes(item):
	return ' '.join(cls for cls, is_present in item.items() if is_present)

def predict(model, args):
	sample = pd.read_csv('../data/sample_submission.csv')
	test_image_path = Path('../data/test/')
	test_dataset = TTADataset(test_image_path, sample, test_transform, 4)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=6)
	
	model.eval()
	all_outputs, all_ids = [], []
	with torch.no_grad():
		for images, image_ids in test_loader:
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
	
	sample = pd.read_csv('../data/sample_submission.csv', index_col='id')

	df = mean_df(df)
	df = df.reindex(sample.index)
	
	out = binarize_prediction(df.values, args.threshold)
	df[:] = out
	df = df.apply(get_classes, axis=1)
	df.name = 'attribute_ids'
	
	df.to_csv(f'submission_{args.model}_{args.threshold}.csv', header=True)

def main(args):
	folds = pd.read_csv('./folds.csv')

	train_folds = folds[folds['fold'] != args.fold]
	valid_folds = folds[folds['fold'] == args.fold]

	train_loader = make_loader(train_folds, train_transform, args)
	valid_loader = make_loader(valid_folds, test_transform, args)


	model = getattr(models, args.model)(pretrained=True, num_classes=N_CLASSES).to(device)
	# if args.pretrained:
	# 	load(model, f'./savings/{model.name}/best_model.pt')
	

	fresh_params = list(model.fresh_params())
	all_params = list(model.parameters())

	if args.mode == 'train':
		train(model, train_loader, valid_loader, fresh_params, args, fresh=True)
		train(model, train_loader, valid_loader, all_params, args)
	elif args.mode == 'validate':
		metrics = validate(model, valid_loader)
		pprint(metrics)
	elif args.mode == 'predict':
		predict(model, args)


if __name__ =='__main__':
	seed_torch()
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('mode', choices = ['train', 'validate', 'predict'])
	arg('--model', default='resnet50')
	arg('--pretrained', type=int, default=1)
	arg('--batch_size', type=int, default=32)
	arg('--threshold', type=float, default=0.5)
	arg('--fold', type=int, default = 0)

	args = parser.parse_args()
	
	main(args)