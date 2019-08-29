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
import os
import sys
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy.misc

from tqdm import tqdm
import argparse

# Define argument
parser = argparse.ArgumentParser()
parser.add_argument('--net', dest='net',default='densenet',type=str,help='choose between [squeezenet,densenet,resnet,combinenet]')
parser.add_argument('--n_grid', dest='n_grid',default=5,type=int,help='size of the subhalo density map')
parser.add_argument('--train_final_layer', dest='train_final_layer',default=True,type=bool,help='whether or not to train the final layer of each model')
args = parser.parse_args()

#-------------------------------------------------------------------------------------------
np.random.seed(999)
TRAINING_SAMPLES = 50000
TESTING_SAMPLES = 10000

folder = '/media/joshua/38B6C6E7B6C6A4AA/v0710/'
model_path = './saved_model_II/'
glo_batch_size = 10
test_num_batch = 50
N_GRID = 56
n_grid = 5
n_subhalos = N_GRID**2

show_test_imgs = False
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            ])
target_transform = torch.Tensor

###
#  Dihedral transformation of bbox (should be in range [0,1] for each label)
###

def dihedral_transform(img,action:0):

    if action == 0:
        pass
    elif action == 1:
        img = np.flip(img,axis=0)
    elif action == 2:
        img = np.flip(img,axis=1)
    elif action == 3:
        img = np.flip(np.flip(img,axis=0),axis=1)
    elif action == 4:
        img = np.flip(np.flip(img,axis=1),axis=0)
    elif action == 5:
        img = np.rot90(img,k=1)
    elif action == 6:
        img = np.rot90(img,k=2)
    elif action == 7:
        img = np.rot90(img,k=3)
    else:
        print("Dihedral group action should be in range [0,7]")

    return img

###
#  Corp Image By four cornering actions (Upper Left, Upper Right, Lower Left, Lower Right)
###
def corp_img(img, labels, corp_ratio=3./4, corner=0):
    img_h, img_w = img.shape
    img_corp_h, img_corp_w = int(img_h*corp_ratio), int(img_w*corp_ratio)

    if corner == 0: # upper left
        img = cv2.resize(img[:img_corp_h,:img_corp_w],(img_h, img_w))

    elif corner == 1: # lower right
        img = cv2.resize(img[img_h-img_corp_h:,img_w-img_corp_w:],(img_h, img_w))


    elif corner == 2: # upper right
        img = cv2.resize(img[:img_corp_h, img_w-img_corp_w:],(img_h, img_w))


    elif corner == 3: # lower left
        img = cv2.resize(img[img_h-img_corp_h:,:img_corp_w],(img_h, img_w))

    else:
        print("Unknown Corp Operation")

    return img


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

    grid_shape = G[y_t-n//2:y_t+1+n//2,x_t-n//2:x_t+1+n//2].shape
    G[y_t-n//2:y_t+1+n//2,x_t-n//2:x_t+1+n//2] = g[:grid_shape[0],:grid_shape[1]]

    return G

def get_noisy_image(image_sub):
    image_sub = image_sub
    background_rms = np.random.uniform(5,20)  # background rms value
    sigma_n = np.random.uniform(0,5)
    gaussian_noise = np.random.randn(224, 224) * sigma_n
    image_sub_noisy = image_sub + gaussian_noise
    image_sub_noisy = np.asarray(image_sub_noisy)
    return image_sub_noisy



def get_PSF(img, PSF_sigma =3.0):
    # 1.0 correspond to pix_res= 0.02 arcsec
    PSF_image = gaussian_filter(img, PSF_sigma)
    return PSF_image

def get_gaussian_noise(img, sigma_n = 10):
    gaussian_noise = np.random.randn(img.shape[0], img.shape[1])
    Noisy_image = img + gaussian_noise
    return Noisy_image


def get_poission_noise(img, peak):
    Noisy_image = np.random.poisson(img/peak) / peak
    return Noisy_image

# 1. use grid target
class DMSDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, n_grid, dataset_index,train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.n_grid = n_grid
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_folder = 'train_'+str(dataset_index)
        self.test_folder = 'test_'+str(dataset_index)

        if self.train:
            self.path = os.path.join(self.root_dir, self.train_folder)
            self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.length = TESTING_SAMPLES


    def __getitem__(self, index):


        # Code for getting training image


        img_path = self.path + '/output_lens_%07d.npy' % (index+1)
        img = np.load(img_path)
        PSF_sigma = np.random.uniform(0, 5)
        img = get_PSF(img, PSF_sigma)
        img = get_noisy_image(img)

        """Code for getting detectable subhalo Kapp Map
        """
        target_subhalo_kappa_density_path = self.path + '/detectable_subhalo_kappa_log10_%07d.npy' % (index+1)
        temp = np.load(target_subhalo_kappa_density_path)

        # eliminate peak value
        temp_flat = temp.flatten()
        temp_flat.sort()
        ind = np.unravel_index(np.argmax(temp, axis =None), temp.shape)
        temp[ind] = temp_flat[-2]

        temp = np.power(10,temp)
        target_subhalo_mass = cv2.resize(temp,(N_GRID,N_GRID),interpolation=cv2.INTER_AREA)

        has_subhalo = 1.0
        target_subhalo_mass[np.isinf(target_subhalo_mass)] = 0.

        """Code for getting detectable subhalo positions
        """
        detectable_num_subhalo_path = self.path + '/detectable_num_subhalo_%07d.npy' % (index+1)
        detectable_num_subhalo = np.load(detectable_num_subhalo_path)
        #print("detectable_num_subhalo", detectable_num_subhalo)

        target_subhalo_density_path = self.path + '/detectable_pos_sub_%07d.npy' % (index+1)
        target_subhalo_density_orig = np.load(target_subhalo_density_path)

        target_subhalo_density = np.zeros((N_GRID, N_GRID))

        if detectable_num_subhalo != 0:
            # only one subhalo
            if len(target_subhalo_density_orig.shape) == 1:
                if len(target_subhalo_density_orig) == 2:
                    density = target2density(*target_subhalo_density_orig , N_GRID, self.n_grid)
                    target_subhalo_density += density
            # multiple subhalos, and detectable subhalos are placed in the first few items
            else:
                for i in range(len(target_subhalo_density_orig)):
                    #     y_t, x_t = (y_t/0.02)/112, (x_t/0.02)/112
                    if len(target_subhalo_density_orig[i]) == 2:
                        density = target2density(*target_subhalo_density_orig[i], N_GRID, self.n_grid)
                        target_subhalo_density += density

            target_subhalo_density = np.clip(target_subhalo_density,0,1)



        img = Image.fromarray(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img.copy())

        if self.target_transform is not None:
            target_subhalo = self.target_transform(target_subhalo_density.copy())
            target_subhalo_mass = self.target_transform(target_subhalo_mass.copy())
            has_subhalo = self.target_transform(np.asarray(has_subhalo))

        return img, target_subhalo, target_subhalo_mass, has_subhalo

    def __len__(self):
        return self.length

class CombineNet(nn.Module):

    def __init__(self, num_models=3):
        super(CombineNet, self).__init__()
        self.combine_conv_macro = nn.Conv2d(num_models, 1, kernel_size=1)
        self.combine_conv_subhalo = nn.Conv2d(num_models, 1, kernel_size=1)
        self.combine_conv_source = nn.Conv2d(num_models, 1, kernel_size=1)

    def forward(self, x):
        combined_macro, combined_subhalo, combined_source = self.combine_conv_macro(x[0]).squeeze(), self.combine_conv_subhalo(x[1]).squeeze(), self.combine_conv_subhalo(x[2]).squeeze()
        return combined_macro, combined_subhalo, combined_source

#-------------------------------------------------------------------------------------------
if __name__ == '__main__':

    import os

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_names = ['squeezenet','densenet','resnet']
    models = {'squeezenet':squeezenet.squeezenetDMSV2, 'densenet':densenet.densenetDMSV2,'resnet': resnet.resnetDMSV2}
    save_fils = {'squeezenet':'squeezeDMSV2','densenet':'densenetDMSV2','resnet': 'resnetDMSV2'}
    n_grid = args.n_grid

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # number of output features
    macro_lensing = 8
    micro_lensing = [1, N_GRID, N_GRID] # the same output feature size applies to source recon as well

    if args.net != "combinenet":
        net = models[args.net](pretrained=False,num_classes=macro_lensing,num_classes_=micro_lensing)
        print("Total number of trainable parameters: ", (count_parameters(net)))

    else:
        net = CombineNet(num_models=3)
        nets = [0]*3
        for i in range(3):
            if args.train_final_layer:
                nets[i] = torch.load(model_path+save_fils[model_names[i]] + '_' + str(n_grid) + '_combine')
            else:
                nets[i] = torch.load(model_path+save_fils[model_names[i]] + '_' + str(n_grid))

            nets[i].cuda()
            for param in nets[i].parameters():
                param.requires_grad = False

            if args.train_final_layer:
                for param in nets[i].classifier.parameters():
                    param.requires_grad = True

                for param in nets[i].classifier_.parameters():
                    param.requires_grad = True

                for param in nets[i].classifier_recon.parameters():
                    param.requires_grad = True
        print("Total number of trainable parameters: ", (count_parameters(net)+sum([count_parameters(model) for model in nets])))

    loss_KL = nn.KLDivLoss(reduction='none')
    loss_mse = nn.MSELoss(reduction='none')
    loss_mae = nn.SmoothL1Loss(reduction='none')
    loss_bce = nn.BCEWithLogitsLoss(reduction='none')

    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = 1e-4)
    best_accuracy = float("inf")

    for index in range(1):

        if not os.path.exists('/media/joshua/38B6C6E7B6C6A4AA/v0710/train_'+str(index)):
            continue

        print('Traning on dataset #',str(index))

        for epoch in range(20):

            net.train()
            total_loss = 0.0
            total_counter = 0
            total_rms = 0
            total_rms_subhalo = 0
            output_macro,output_subhalo,output_source = 0,0,0

            train_loader = torch.utils.data.DataLoader(DMSDataset(folder, n_grid, dataset_index=index,train=True, transform=data_transform, target_transform=target_transform),
                                batch_size = glo_batch_size, shuffle = True
                                )

            for batch_idx, (data, target_subhalo, target_subhalo_mass, has_subhalo) in enumerate(tqdm(train_loader, total = len(train_loader))):
                data = Variable(data).cuda()
                target_subhalo = Variable(target_subhalo).cuda()
                target_subhalo_mass = Variable(target_subhalo_mass).cuda()
                has_subhalo = Variable(has_subhalo).cuda()

                optimizer.zero_grad()

                if args.net != "combinenet":
                    X = data
                else:
                    output_macro_list = [0]*3
                    output_subhalo_list = [0]*3
                    output_source_list = [0]*3
                    for i in range(3):
                        output_macro_list[i],output_subhalo_list[i],output_source_list[i]  = nets[i](data)
                        output_macro_list[i] = output_macro_list[i].unsqueeze(1).unsqueeze(2)
                        output_subhalo_list[i] = output_subhalo_list[i].unsqueeze(1)
                        output_source_list[i] = output_source_list[i].unsqueeze(1)
                    X = [0]*3
                    X[0] = torch.cat(output_macro_list,dim=1)
                    X[1] = torch.cat(output_subhalo_list,dim=1)
                    X[2] = torch.cat(output_source_list,dim=1)

                output_macro,output_subhalo,output_source = net(X)

                # Calculate Losses
                loss_subhalo = loss_bce(output_subhalo.unsqueeze(3), target_subhalo.unsqueeze(3))

                loss = torch.mean(loss_subhalo)

                total_loss += loss.item()
                total_counter += 1

                loss.backward()
                optimizer.step()

            print(epoch, 'Train loss (averge per batch wise):', total_loss/(total_counter))
            try:
                with torch.no_grad():
                    net.eval()
                    total_loss = 0.0
                    total_counter = 0
                    total_rms = 0
                    total_rms_subhalo = 0

                    test_loader = torch.utils.data.DataLoader(DMSDataset(folder, n_grid,dataset_index=index, train=False, transform=data_transform, target_transform=target_transform),
                                batch_size = glo_batch_size, shuffle = True
                                )

                    for batch_idx,(data, target_subhalo, target_subhalo_mass, has_subhalo)in enumerate(test_loader):
                        data = Variable(data).cuda()
                        target_subhalo = Variable(target_subhalo).cuda()
                        target_subhalo_mass = Variable(target_subhalo_mass).cuda()
                        has_subhalo = Variable(has_subhalo).cuda()

                        if args.net != "combinenet":
                            X = data
                        else:
                            output_macro_list = [0]*3
                            output_subhalo_list = [0]*3
                            output_source_list = [0]*3
                            for i in range(3):
                                output_macro_list[i],output_subhalo_list[i],output_source_list[i]  = nets[i](data)
                                output_macro_list[i] = output_macro_list[i].unsqueeze(1).unsqueeze(2)
                                output_subhalo_list[i] = output_subhalo_list[i].unsqueeze(1)
                                output_source_list[i] = output_source_list[i].unsqueeze(1)

                            X = [0]*3
                            X[0] = torch.cat(output_macro_list,dim=1)
                            X[1] = torch.cat(output_subhalo_list,dim=1)
                            X[2] = torch.cat(output_source_list,dim=1)

                        output_macro,output_subhalo,output_source = net(X)

                        # Calculate Losses

                        loss_subhalo = loss_bce(output_subhalo.unsqueeze(3), target_subhalo.unsqueeze(3))


                        loss_subhalo_mass = loss_mae(output_source.unsqueeze(3), target_subhalo_mass.unsqueeze(3))


                        loss = torch.mean(loss_subhalo) #+ torch.mean(loss_subhalo_mass) #+ loss_macro_has_subhalo


                        total_loss += loss.item()
                        total_counter += 1

                        if batch_idx % test_num_batch == 0 and batch_idx != 0:
                            break


                    print(epoch, 'Test loss (averge per batch wise):', total_loss/(total_counter))
            except:
                pass

            if total_loss/(total_counter) < best_accuracy:
                best_accuracy = total_loss/(total_counter)
                torch.save(net, model_path + args.net + '_' + str(n_grid))

                if args.net == "combinenet":
                    for i in range(len(nets)):
                        torch.save(nets[i], model_path + save_fils[model_names[i]] + '_' + str(n_grid) + '_combine')

                print("saved to file.")
