import os
import json
import torch
import cv2
import math
import numpy as np
from pathlib import Path
import torch.utils.data as Data
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image



def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class YouTubeVOS_Data(Dataset):
	def __init__(self, dataset_folder=None, transforms_train=None, transform_test =None):
		self.dataset_folder = Path("C:/Users/Siyao/Downloads/YOUTUBE-VOS/train_zip/train")
		self.JPEGPath = str(os.path.join(self.dataset_folder,'JPEGImages'))
		self.AnnPath = str(os.path.join(self.dataset_folder,'Annotations'))
		self.json_path = str(os.path.join(self.dataset_folder,'meta.json')) 
		self.transforms = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize([256,448]),
                                    # transforms.ColorJitter(0.4,0.4,0.4),
                                    # transforms.GaussianBlur(3,(0.1,5)),
                                    transforms.ToTensor()
                                ])
		self.first_frame = True
		self.transform_test = transform_test
		self.folders = os.listdir(self.JPEGPath)
		with open(self.json_path) as file:
   			self.json_file = json.load(file)

		
	def __getitem__(self,index):
		data_dict = {}
		folder_name = self.folders[index]
		object_list = (list(self.json_file['videos'][folder_name]['objects'].keys()))	# List of objects in current video
		frameid_list = self.json_file['videos'][folder_name]['objects'][object_list[0]]['frames']
		image_list = []
		mask_list = []
		for frame_num in frameid_list:
			image_path = self.JPEGPath + '/' + folder_name + '/' + frame_num + '.jpg'
			mask_path = self.AnnPath + '/' + folder_name + '/' + frame_num + '.png'
			image_array = load_image(image_path)
			# image_tensor = torch.from_numpy(image_array)
			image_tensor = self.transforms(image_array)
			image_list.append(image_tensor)
			mask_array = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)	# Convert mask from RGB to 1D
			mask_tensor = torch.from_numpy(mask_array)
			mask_list.append(mask_tensor)
			
		frames_stack = torch.stack(image_list, dim=0)
		masks_stack = torch.stack(mask_list, dim=0)
		pres_stack = torch.zeros(masks_stack.shape)
		if self.first_frame == True:
			pres_stack[0] = mask_list[0]
		pres_stack = F.interpolate(pres_stack.unsqueeze(1),size=[256,448],mode='nearest').squeeze(1)

		data_dict['Frames'] = frames_stack
		data_dict['Masks'] = masks_stack
		data_dict['Preds'] = pres_stack
		data_dict['Num_objs'] = int(torch.max(masks_stack))
		return data_dict


	def __len__(self): 
		return len(self.folders)