import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

class KittiDataset(data.Dataset):
    r""" Kitti Instance dataset.

    Args:
        dataset_folder (str): path to the KITTI dataset
        img_size (int): size of the cropped images
        transform (list): list of transformations applied to the images
        return_idx (bool): wether to return index
    """

    def __init__(self, dataset_folder, img_size=224, transform=None, return_idx=False):
        self.img_size = img_size
        self.img_path = os.path.join(dataset_folder, 'image_2')
        crop_path = os.path.join(dataset_folder, 'cropped_images')
        self.cropped_images = []
        for folder in os.listdir(crop_path):
            folder_path = os.path.join(crop_path, folder)
            for file_name in os.listdir(folder_path):
                current_file_path = os.path.join(folder_path, file_name)
                self.cropped_images.append(current_file_path)

        self.len = len(self.cropped_images)
        self.transform = transform
        self.return_idx = return_idx
    
    def get_model_dict(self, idx):
        model_dict = {
            'model': str(idx),
            'category': 'kitti',
        }
        return model_dict

    def get_model(self, idx):
        ''' Returns the model.

        Args:
            idx (int): ID of data point
        '''
        f_name = os.path.basename(self.cropped_images[idx])[:-4]
        return f_name

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return self.len

    def __getitem__(self, idx):
        ''' Returns the data point.

        Args:
            idx (int): ID of data point
        '''
        ori_file_name = os.path.basename(self.cropped_images[idx])[:9] + '.png'
        original_img = Image.open(os.path.join(self.img_path, ori_file_name))
        cropped_img = Image.open(self.cropped_images[idx])

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        original_img = transforms.ToTensor()(original_img)
        cropped_img = transform(cropped_img)
        idx = torch.tensor(idx)

        data = {
            'inputs': cropped_img,
            'idx': idx,
        }

        return data


class OnlineProductDataset(data.Dataset):
    r""" Stanford Online Product Dataset.

    Args:
        dataset_folder (str): path to the dataset dataset
        img_size (int): size of the cropped images
        classes (list): list of classes
        max_number_imgs (int): maximum number of images
        return_idx (bool): wether to return index
        return_category (bool): wether to return category
    """

    def __init__(self, dataset_folder, img_size=224, classes=['chair'],
                 max_number_imgs=1000, return_idx=False, return_category=False):

        self.img_size = img_size
        self.dataset_folder = dataset_folder
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        self.class_id = {}
        self.metadata = []

        for i, cl in enumerate(classes):
            self.metadata.append({'name': cl})
            self.class_id[cl] = i
            cl_names = np.loadtxt(
                os.path.join(dataset_folder, cl+'_final.txt'), dtype=np.str)
            cl_names = cl_names[:max_number_imgs]
            att = np.vstack(
                (cl_names, np.full_like(cl_names, cl))).transpose(1, 0)
            if i > 0:
                self.file_names = np.vstack((self.file_names, att))
            else:
                self.file_names = att

        self.len = self.file_names.shape[0]
        self.return_idx = return_idx
        self.return_category = return_category

    def get_model_dict(self, idx):
        category_id = self.class_id[self.file_names[idx, 1]]

        model_dict = {
            'model': str(idx),
            'category': category_id
        }
        return model_dict


    def get_model(self, idx):
        ''' Returns the model.

        Args:
            idx (int): ID of data point
        '''
        file_name = os.path.basename(self.file_names[idx, 0])[:-4]
        return file_name

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return self.len

    def __getitem__(self, idx):
        ''' Returns the data point.

        Args:
            idx (int): ID of data point
        '''
        f = os.path.join(
            self.dataset_folder,
            self.file_names[idx, 1]+'_final',
            self.file_names[idx, 0])

        img_in = Image.open(f)
        img = Image.new("RGB", img_in.size)
        img.paste(img_in)
        idx = torch.tensor(idx)
        cl_id = torch.tensor(self.class_id[self.file_names[idx, 1]])

        if self.transform:
            img = self.transform(img)

        data = {
            'inputs': img,
        }

        if self.return_idx:
            data['idx'] = idx

        if self.return_category:
            data['category'] = cl_id

        return data


IMAGE_EXTENSIONS = (
    '.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'
)

class ImageDataset(data.Dataset):
    r""" Cars Dataset.

    Args:
        dataset_folder (str): path to the dataset dataset
        img_size (int): size of the cropped images
        transform (list): list of transformations applied to the data points
    """

    def __init__(self, dataset_folder, img_size=224, transform=None, return_idx=False):
        """

        Arguments:
            dataset_folder (path): path to the KITTI dataset
            img_size (int): required size of the cropped images
            return_idx (bool): wether to return index
        """

        self.img_size = img_size
        self.img_path = dataset_folder
        self.file_list = os.listdir(self.img_path)
        self.file_list = [
            f for f in self.file_list 
            if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
        ]
        self.len = len(self.file_list)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.return_idx = return_idx

    def get_model(self, idx):
        ''' Returns the model.

        Args:
            idx (int): ID of data point
        '''
        f_name = os.path.basename(self.file_list[idx])
        f_name = os.path.splitext(f_name)[0]
        return f_name

    def get_model_dict(self, idx):  
        f_name = os.path.basename(self.file_list[idx])
        model_dict = {
            'model': f_name
        }
        return model_dict
        
    def __len__(self):
        ''' Returns the length of the dataset.'''
        return self.len

    def __getitem__(self, idx):
        ''' Returns the data point.

        Args:
            idx (int): ID of data point
        '''
        f = os.path.join(self.img_path, self.file_list[idx])
        img_in = Image.open(f)
        img = Image.new("RGB", img_in.size)
        img.paste(img_in)
        if self.transform:
            img = self.transform(img)

        idx = torch.tensor(idx)

        data = {
            'inputs': img,
        }

        if self.return_idx:
            data['idx'] = idx

        return data
