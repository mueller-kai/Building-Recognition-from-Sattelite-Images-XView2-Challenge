if __name__ == '__main__': 
	import dataset
	import config
	from model import UNet
	#import config
	from torch.nn import BCEWithLogitsLoss, BCELoss
	from torch.optim import Adam
	from torch.utils.data import DataLoader, dataloader
	from torchvision import transforms
	from imutils import paths
	from tqdm import tqdm
	import matplotlib.pyplot as plt
	import torch
	import time
	import os
	import cv2

	transforms = transforms.Compose([transforms.ToTensor()])
	#transforms.ToPILImage(),transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),

	destaster_vision_dataset = dataset.DestasterVisionDataset(
		image_folder='train/images',
		target_folder='train/targets',
		labels_folder='train/labels',
		transforms=transforms
	)

	destaster_vision_validationset = dataset.DestasterVisionDataset(
		image_folder='validation/images',
		target_folder='validation/targets',
		labels_folder='validation/labels',
		transforms= transforms
	)

	# load the image and mask filepaths in a sorted manner
	imagePaths = destaster_vision_dataset.images_paths_pre
	targetPaths = destaster_vision_dataset.target_paths_pre

	trainloader = DataLoader(destaster_vision_dataset, shuffle=False,
		batch_size= config.BATCH_SIZE, drop_last=True, pin_memory=False,
		num_workers=os.cpu_count())

	validationloader = DataLoader(destaster_vision_validationset, shuffle=False,
		batch_size= config.BATCH_SIZE, drop_last=True, pin_memory=False,
		num_workers=os.cpu_count())

	# initialize our UNet model
	unet = UNet().to(config.DEVICE)
	lossFunc = BCEWithLogitsLoss()
	optimizer = Adam(unet.parameters(), lr=config.INIT_LR)

	#calculate needed training steps
	trainSteps = len(trainloader.dataset.images_paths_pre) // config.BATCH_SIZE
	validationSteps = len(validationloader.dataset.images_paths_pre) // config.BATCH_SIZE

	#create empt dict to store history
	history = {'train_loss': [], 'validation_loss': []}

	#loop over epochs
	print("[INFO] starting training")
	startTime = time.time()
	for e in tqdm(range(config.NUM_EPOCHS)):
		print('epoch started')
		# set the model in training mode
		unet.train()

		# initialize the total training and validation loss
		totalTrainLoss = 0
		totalValidationLoss = 0
		currentstep = 0
		# loop over the training set
		for (i, (x, y)) in enumerate(trainloader):

			#debug print statements
			#print(f'cuurent step: {currentstep} of {trainSteps} steps')
			#print currently used picture names
			#print(f"i: {i}")
			#print(trainloader.dataset.images_paths_pre[i])
			#print(trainloader.dataset.target_paths_pre[i])
			# send the input to the device

			print(f"i:{i} ", end = "", flush=True)
			
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
			totalTrainLoss += loss.item()
			currentstep += 1

		print("switching off autograd")
		# switch off autograd
		with torch.no_grad():
			# set the model in evaluation mode
			unet.eval()
			# loop over the validation set
			for (x, y) in validationloader:
				# send the input to the device
				(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
				# make the predictions and calculate the validation loss
				pred = unet(x)
				totalValidationLoss += lossFunc(pred, y).item()
		
		# calculate the average training and validation loss
		print("calculating losses")
		avgTrainLoss = totalTrainLoss / trainSteps
		avgValidationLoss = totalValidationLoss / validationSteps
		# update our training history
		history["train_loss"].append(avgTrainLoss)
		history["validation_loss"].append(avgValidationLoss)
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
		print("Train loss: {:.6f}, Validation loss: {:.4f}".format(
			avgTrainLoss, avgValidationLoss))

		#save model every 10 Epoch
		if e % 10 == 0:
			print("saved‚")
			torch.save(unet, config.MODEL_PATH + f"e{e}_.pth")

			# plot the training loss
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(history["train_loss"], label="train_loss")
			plt.plot(history["validation_loss"], label="validation_loss")
			plt.title("Training Loss on Dataset")
			plt.xlabel("Epoch")
			plt.ylabel("Loss")
			plt.legend(loc="lower left")
			plt.savefig(f'output/lossafterDataLoaderFix.png')

	# display the total time needed to perform the training
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))

	# serialize the model to disk
	torch.save(unet, config.MODEL_PATH)