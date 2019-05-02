

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

if __name__ == '__main__':
	pass