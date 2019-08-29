#### Train_script by Hang Yu, Joshua Yao-Yu Lin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from  torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import model.squeezenet as squeezenet
import model.densenet as densenet
import model.resnet as resnet
import lenstronomy.Util.image_util as image_util

# torch.set_default_tensor_type('torch.cuda.FloatTensor')


import os
import sys
import pandas as pd
from tensorboardX import SummaryWriter
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import argparse

# Define argument
parser = argparse.ArgumentParser()
parser.add_argument('--net', dest='net',default='dense',type=str,help='choose between squeeze,dense')
args = parser.parse_args()


#-------------------------------------------------------------------------------------------
np.random.seed(999)
TRAINING_SAMPLES = 20000
TESTING_SAMPLES = 2000

folder = '/home/joshua/Documents/git_work_zone/Dark_Matter_Sub/v1103'
glo_batch_size = 32
test_num_batch = 50
N_GRID = 56
n_grid = 5
n_subhalos = N_GRID**2

show_test_imgs = False
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            normalize,
            ])
target_transform = torch.Tensor

###
#  Dihedral transformation of bbox (should be in range [0,1] for each label)
###

def dihedral_transform(img,labels,action:0):

    if action == 0:
        pass
    elif action == 1:
        img = np.flip(img,axis=0)
        labels = np.flip(labels,axis=0)
    elif action == 2:
        img = np.flip(img,axis=1)
        labels = np.flip(labels,axis=1)
    elif action == 3:
        img = np.flip(np.flip(img,axis=0),axis=1)
        labels = np.flip(np.flip(labels,axis=0),axis=1)
    elif action == 4:
        img = np.flip(np.flip(img,axis=1),axis=0)
        labels = np.flip(np.flip(labels,axis=1),axis=0)
    elif action == 5:
        img = np.rot90(img,k=1)
        labels = np.rot90(labels,k=1)
    elif action == 6:
        img = np.rot90(img,k=2)
        labels = np.rot90(labels,k=2)
    elif action == 7:
        img = np.rot90(img,k=3)
        labels = np.rot90(labels,k=3)
    else:
        print("Dihedral group action should be in range [0,7]")

    return img, labels

###
#  Corp Image By four cornering actions (Upper Left, Upper Right, Lower Left, Lower Right)
###
def corp_img(img, labels, corp_ratio=3./4, corner=0):
    img_h, img_w = img.shape
    img_corp_h, img_corp_w = int(img_h*corp_ratio), int(img_w*corp_ratio)

    label_h, label_w = labels.shape
    label_corp_h, label_corp_w = int(label_h*corp_ratio), int(label_w*corp_ratio)

    if corner == 0: # upper left
        img = cv2.resize(img[:img_corp_h,:img_corp_w],(img_h, img_w))
        labels = cv2.resize(labels[:label_corp_h,:label_corp_w],(label_h, label_w))

    elif corner == 1: # lower right
        img = cv2.resize(img[img_h-img_corp_h:,img_w-img_corp_w:],(img_h, img_w))
        labels = cv2.resize(labels[label_h-label_corp_h:,label_w-label_corp_w:],(label_h, label_w))


    elif corner == 2: # upper right
        img = cv2.resize(img[:img_corp_h, img_w-img_corp_w:],(img_h, img_w))
        labels = cv2.resize(labels[:label_corp_h, label_w-label_corp_w:],(label_h, label_w))


    elif corner == 3: # lower left
        img = cv2.resize(img[img_h-img_corp_h:,:img_corp_w],(img_h, img_w))
        labels = cv2.resize(labels[label_h-label_corp_h:,:label_corp_w],(label_h, label_w))
    else:
        print("Unknown Corp Operation")

    return img, labels


# This function create a N_GRID x N_GRID zeros matrix and insert a n_grid x n_grid gaussian distribution
# in the (x_t, y_t), notice n_grid must be an odd number.
def target2density(x_t,y_t,N_GRID,n_grid):
    N, n = N_GRID,n_grid
    x, y = np.meshgrid(np.linspace(-1,1,n), np.linspace(-1,1,n))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

    G = np.zeros((N,N))
    y_t = int(N/2 + y_t*N/2)
    x_t = int(N/2 + x_t*N/2)

    G[y_t-n//2:y_t+1+n//2,x_t-n//2:x_t+1+n//2] = g
    return G

def get_noisy_image(image_sub):
    image_sub = image_sub
    exp_time = 100  # exposure time to quantify the Poisson noise level
    background_rms = np.random.uniform(5,20)  # background rms value
    poisson = image_util.add_poisson(image_sub, exp_time=exp_time)
    bkg = image_util.add_background(image_sub, sigma_bkd=background_rms)
    image_sub_noisy = image_sub + bkg + poisson
    image_sub_noisy = np.asarray(image_sub_noisy)
    return image_sub_noisy

# 1. use grid target
class DMSDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_folder = 'data_train'
        self.test_folder = 'data_test'

        if self.train:
            self.path = os.path.join(self.root_dir, self.train_folder)
            self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.length = TESTING_SAMPLES

    def __getitem__(self, index):

        df = pd.read_csv(self.path + '/lens_sub_%07d.txt' % (index+1), header=None)
        A = df.values
        # hard code final target features
        target_macro = A[0]
        target_subhalo = A[1:,:-3]
        # target_subhalo has shape (n_subhalo,5)

        # hard code final output features
        target_subhalo_density = np.zeros((N_GRID, N_GRID))
        for i in range(len(target_subhalo)):
            if target_subhalo[i,-1] != 0:
                target_subhalo_density += target2density(*target_subhalo[i,2:4], N_GRID, n_grid)

        img_path = self.path + '/lens_sub_%07d.png' % (index+1)
        # Resize images and apply noise to them
        img = np.asarray(Image.open(img_path).convert("L").resize((224,224)))

        img,target_subhalo_density = corp_img(img,target_subhalo_density,corner=np.random.randint(4))
        img,target_subhalo_density = dihedral_transform(img,target_subhalo_density,action=np.random.randint(8))

        img = Image.fromarray(get_noisy_image(img)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img.copy())

        if self.target_transform is not None:
            target_macro = self.target_transform(target_macro.copy())
            target_subhalo = self.target_transform(target_subhalo_density.copy())

        return img, target_macro, target_subhalo

    def __len__(self):
        return self.length

# Important: Change datasets name correctly. 'v0531' stands for a small dataset generated on 5/31/2018
train_loader = torch.utils.data.DataLoader(DMSDataset(folder, train=True, transform=data_transform, target_transform=target_transform),
                    batch_size = glo_batch_size, shuffle = True
                    )

#-------------------------------------------------------------------------------------------
if __name__ == '__main__':

    root_folder = './saved_model/'
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)

    models = {'squeeze':squeezenet.squeezenetDMS, 'dense':densenet.densenetDMS,'resnet': resnet.resnetDMS}
    save_fils = {'squeeze':'squeeze11','dense':'densenetDMS','resnet': 'resnetDMS'}

    # hard code final output features
    macro_lensing = 8
    micro_lensing = [1, N_GRID, N_GRID]

    net = models[args.net](pretrained=False,num_classes=macro_lensing,num_classes_=micro_lensing)
    loss_fn = nn.MSELoss(reduce=False)
    loss_fn2 = nn.BCEWithLogitsLoss(reduce=False)

    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr = 1e-4)

    best_accuracy = float("inf")

    #if os.path.exists('./saved_model/densenetDMS'):
        #net = torch.load('./saved_model/densenetDMS')
        #print('loaded mdl!')

    for epoch in range(10):

        net.train()
        total_loss = 0.0
        total_counter = 0
        total_rms = 0
        total_rms_subhalo = 0

        for batch_idx, (data, target_macro, target_subhalo) in enumerate(tqdm(train_loader, total = len(train_loader))):
            data, target_macro, target_subhalo = Variable(data).cuda(), Variable(target_macro).cuda(), Variable(target_subhalo).cuda()

            optimizer.zero_grad()
            output = net(data)

            output_macro = output[0]
            output_subhalo = output[1]

            loss_subhalo = loss_fn2(output_subhalo.unsqueeze(3), target_subhalo.unsqueeze(3))
            loss_subhalo = torch.mean(torch.mean(torch.mean(loss_subhalo,dim=3),dim=2),dim=1)

            loss = torch.mean(loss_fn(output_macro[:,:-1], target_macro[:,:-1])) + \
                   torch.mean(loss_fn2(output_macro[:,-1], target_macro[:,-1]))
            loss += torch.mean(target_macro[:,-1].unsqueeze(1) * loss_subhalo)

            square_diff = (output_macro[:,:-1] - target_macro[:,:-1])**2
            accuracy = ((torch.abs(F.sigmoid(output_macro[:,-1].unsqueeze(1)) - target_macro[:,-1].unsqueeze(1))<=0.5).sum()).item()
            total_rms += np.append(torch.sqrt(torch.mean(square_diff,dim=0)).detach().cpu().numpy(),accuracy / glo_batch_size)

            total_loss += loss.item()
            total_counter += 1

            loss.backward()
            optimizer.step()

        # Collect RMS over each label
        avg_rms = total_rms / (total_counter)

        # print test loss and tets rms
        print(epoch, 'Train loss (averge per batch wise):', total_loss/(total_counter), ' RMS_Macro (average per batch wise):', np.array_str(avg_rms, precision=3))

        with torch.no_grad():
            net.eval()
            total_loss = 0.0
            total_counter = 0
            total_rms = 0
            total_rms_subhalo = 0

            test_loader = torch.utils.data.DataLoader(DMSDataset(folder, train=False, transform=data_transform, target_transform=target_transform),
                        batch_size = glo_batch_size, shuffle = True
                        )

            for batch_idx, (data, target_macro, target_subhalo) in enumerate(test_loader):
                data, target_macro, target_subhalo = Variable(data).cuda(), Variable(target_macro).cuda(), Variable(target_subhalo).cuda()

                output = net(data)
                output_macro = output[0]
                output_subhalo = output[1]

                loss_subhalo = loss_fn2(output_subhalo.unsqueeze(3), target_subhalo.unsqueeze(3))
                loss_subhalo = torch.mean(torch.mean(torch.mean(loss_subhalo,dim=3),dim=2),dim=1)


                loss = torch.mean(loss_fn(output_macro[:,:-1], target_macro[:,:-1])) + \
                       torch.mean(loss_fn2(output_macro[:,-1], target_macro[:,-1]))
                loss += torch.mean(target_macro[:,-1].unsqueeze(1) * loss_subhalo)

                square_diff = (output_macro[:,:-1] - target_macro[:,:-1])**2
                accuracy = ((torch.abs(F.sigmoid(output_macro[:,-1].unsqueeze(1)) - target_macro[:,-1].unsqueeze(1))<=0.5).sum()).item()
                total_rms += np.append(torch.sqrt(torch.mean(square_diff,dim=0)).detach().cpu().numpy(),accuracy / glo_batch_size)

                total_loss += loss.item()
                total_counter += 1

                if batch_idx % test_num_batch == 0 and batch_idx != 0:
                    break

            # Collect RMS over each label
            avg_rms = total_rms / (total_counter)

            # print test loss and tets rms
            print(epoch, 'Test loss (averge per batch wise):', total_loss/(total_counter), ' RMS_Macro (average per batch wise):', np.array_str(avg_rms, precision=3))

            if total_loss/(total_counter) < best_accuracy:
                best_accuracy = total_loss/(total_counter)
                torch.save(net, './saved_model/' + save_fils[args.net])
                print("saved to file.")

cv2.destroyAllWindows()
