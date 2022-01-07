if __name__ == '__main__': 
	import avrg_pix_diff
	import config
	from model import UNet
	#import config
	from torch.nn import BCEWithLogitsLoss
	from torch.optim import Adam
	from torch.utils.data import DataLoader, dataloader
	from sklearn.model_selection import train_test_split
	from torchvision import transforms
	from imutils import paths
	from tqdm import tqdm
	import matplotlib.pyplot as plt
	import torch
	import time
	import os
	import cv2

	transforms = transforms.Compose([transforms.ToPILImage(),
	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

	destaster_vision_dataset = avrg_pix_diff.DestasterVisionDataset(
		image_folder='images',
		target_folder='targets',
		labels_folder='labels',
		transforms=transforms
	)

	destaster_vision_testset = avrg_pix_diff.DestasterVisionDataset(
		image_folder='test_set/images',
		target_folder='test_set/targets',
		labels_folder='test_set/labels',
		transforms= transforms
	)

	# load the image and mask filepaths in a sorted manner
	imagePaths = destaster_vision_dataset.images_paths_pre
	targetPaths = destaster_vision_dataset.target_paths_pre

	trainloader = DataLoader(destaster_vision_dataset, shuffle=True,
		batch_size= config.BATCH_SIZE, pin_memory=False,
		num_workers=os.cpu_count())

	testloader = DataLoader(destaster_vision_testset, shuffle=True,
		batch_size= config.BATCH_SIZE, pin_memory=False,
		num_workers=os.cpu_count())

	# initialize our UNet model
	unet = UNet().to('cpu')
	lossFunc = BCEWithLogitsLoss()
	optimizer = Adam(unet.parameters(), lr=config.INIT_LR)

	#calculate needed training steps
	trainSteps = destaster_vision_dataset.__len__() // config.BATCH_SIZE
	testSteps = destaster_vision_testset.__len__() // config.BATCH_SIZE

	#create empt dict to store history
	history = {'train': []}

	#loop over epochs
	print("[INFO] starting training")
	startTime = time.time()
	for e in tqdm(range(config.NUM_EPOCHS)):
		print('epoch started')
		# set the model in training mode
		unet.train()

		# initialize the total training and validation loss
		totalTrainLoss = 0
		totalTestLoss = 0

		# loop over the training set
		for (i, (x, y)) in enumerate(trainloader):
			print('ping')
			'''
			#load images
			x = cv2.imread(x[i])
			y = cv2.imread(y[i])

			#turn images to tensor
			x = transforms(x)
			y = transforms(y)
			'''

			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			# perform a forward pass and calculate the training loss
			pred = unet(x)
			loss = lossFunc(pred, y)
			# first, zero out any previously accumulated gradients, then
			# perform backpropagation, and then update model parameters
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# add the loss to the total training loss so far
			totalTrainLoss += loss
		

		# switch off autograd
		with torch.no_grad():
			# set the model in evaluation mode
			unet.eval()
			# loop over the validation set
			for (x, y) in testloader:
				# send the input to the device
				(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
				# make the predictions and calculate the validation loss
				pred = unet(x)
				totalTestLoss += lossFunc(pred, y)
		
		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgTestLoss = totalTestLoss / testSteps
		# update our training history
		history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		history["test_loss"].append(avgTestLoss.cpu().detach().numpy())
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
		print("Train loss: {:.6f}, Test loss: {:.4f}".format(
			avgTrainLoss, avgTestLoss))

	# display the total time needed to perform the training
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))

	# plot the training loss
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H["train_loss"], label="train_loss")
	plt.plot(H["test_loss"], label="test_loss")
	plt.title("Training Loss on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(config.PLOT_PATH)
	# serialize the model to disk
	torch.save(unet, config.MODEL_PATH)