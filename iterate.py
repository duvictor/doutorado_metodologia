# para fazer a gpu funcionar no pytorch
# siga este tutorial https://pub.towardsai.net/installing-pytorch-with-cuda-support-on-windows-10-a38b1134535e
# e finalmente instale pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# criado em 05/06/2022
# a ideia é carregar um exame inteiro para memória do computador e verificar a segmentação do mesmo
# E:\PycharmProjects\pythonProject\exame\CQ500CT257 vai ser o train
# E:\PycharmProjects\pythonProject\exame\CQ500CT420 vai ser o test


# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
# import matplotlib.pyplot as plt
import random
import pydicom

import os
import torch
import pydicom
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset




"""
# Train images: ImageData/train/0; ImageData/train/1
# Test images: ImageData/test/0; ImageData/test/1
# Classes: "not_operable": 0; "operable":1
#
# set_file_matrix():
# Count total items in sub-folders of root/image_dir:
# Create a list with all items from root/image_dir "(pixel_array, label)"
# Takes label from sub-folder name "0" or "1"
"""
class DicomDataset(Dataset):
    def __init__(self, root, image_dir):
        self.image_dir = os.path.join(root, image_dir)  # ImageData/train or ImageData/test
        self.list_IDs = list([f.path for f in os.scandir(self.image_dir) if f.is_dir()])
        self.data = self.set_file_matrix()
        self.transform = transforms.Compose([ transforms.Resize(128), transforms.CenterCrop(128)])

    def __len__(self):
        return len(self.data)

    def normalize_contrast(self, voxel):
        if voxel.sum() == 0:
            return voxel
        voxel = voxel - np.min(voxel)
        voxel = voxel / np.max(voxel)
        voxel = (voxel * 255).astype(np.uint8)
        return voxel

    def set_file_matrix(self):
        # count elements
        total = 0
        root = self.image_dir
        folders = ([name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))])
        for folder in folders:
            new_path = os.path.join(root, folder)
            contents = len([name for name in os.listdir(new_path) if os.path.isfile(os.path.join(new_path, name))])
            total += contents

        # create list(img_name, label)
        files = []
        labels = ([name for name in os.listdir(root) if os.path.isfile(os.path.join(root, name))])
        for label in labels:
            new_path = os.path.join(root, label)
            files.append(new_path)
        return files

    def __getitem__(self, index):
        image_file = pydicom.dcmread(self.data[index])
        image = np.array(image_file.pixel_array, dtype=np.float32)[np.newaxis]  # Add channel dimension

        image2 = self.normalize_contrast(image)
        cv2.imshow("original", image2[0])
        cv2.waitKey(10)

        image = torch.from_numpy(image2)
        # if self.transform:
        #     image = self.transform(image)

        return image



# ROOT_PATH = "E:\\PycharmProjects\\pythonProject\\exame"
# train_set = DicomDataset(ROOT_PATH, 'train')
# test_set = DicomDataset(ROOT_PATH, 'test')

ROOT_PATH = "D:\\dataset\\rsna-miccai-brain-tumor-radiogenomic-classification\\train\\00000"

train_set = DicomDataset(ROOT_PATH, 'T2w')
test_set = DicomDataset(ROOT_PATH, 'FLAIR')
train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)








use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=500, type=int, help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, help='learning rate')
parser.add_argument('--nConv', metavar='M', default=3, type=int, help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, help='visualization flag')
# parser.add_argument('--input', metavar='FILENAME', default=r'D:\Users\paulo\PycharmProjects\pytorch-unsupervised-segmentation-tip\imagens\3.png', help='input image file name', required=False)
parser.add_argument('--input', metavar='FILENAME',
                    default=r'D:\Users\paulo\PycharmProjects\pytorch-unsupervised-segmentation-tip\imagens\Normal-74-yo-CT-head-5_pt.jpg',
                    help='input image file name', required=False)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float, help='step size for similarity loss',
                    required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, help='step size for scribble loss')
args = parser.parse_args()


# path = "E:\\PycharmProjects\\pythonProject\\exame\\CQ500CT420\\Unknown Study\\CT 5mm\\CT000000.dcm"







# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv - 1):
            self.conv2.append(nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(args.nChannel))
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        for i in range(args.nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


# load image
# im = cv2.imread(args.input)
# data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
# if use_cuda:
#     data = data.cuda()
# data = Variable(data)

# load scribble
# if args.scribble:
#     mask = cv2.imread(args.input.replace('.' + args.input.split('.')[-1], '_scribble.png'), -1)
#     mask = mask.reshape(-1)
#     mask_inds = np.unique(mask)
#     mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
#     inds_sim = torch.from_numpy(np.where(mask == 255)[0])
#     inds_scr = torch.from_numpy(np.where(mask != 255)[0])
#     target_scr = torch.from_numpy(mask.astype(np.int))
#     if use_cuda:
#         inds_sim = inds_sim.cuda()
#         inds_scr = inds_scr.cuda()
#         target_scr = target_scr.cuda()
#     target_scr = Variable(target_scr)
#     # set minLabels
#     args.minLabels = len(mask_inds)

for item in enumerate(train_set_loader):
    item = item[1]
    break

# train
model = MyNet(item.size(1))
print(model)
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()
# scribble loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()
# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average=True)
loss_hpz = torch.nn.L1Loss(size_average=True)

HPy_target = torch.zeros(512 - 1, 512, args.nChannel)
HPz_target = torch.zeros(512, 512 - 1, args.nChannel)
if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))



# num_epochs = args.maxIter
num_epochs = 4

# more code before
n_total_steps = len(train_set_loader)
for batch_idx in range(args.maxIter):
    i = 0
    for item in enumerate(train_set_loader):
        item = item[1]
        item = item.float()
        if use_cuda:
            item = item.cuda()
        item = Variable(item)
        # forwarding
        optimizer.zero_grad()
        output = model(item)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

        outputHP = output.reshape((512, 512, args.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)

        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()

        nLabels = len(np.unique(im_target))

        if args.visualize:
            im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(512,512,3).astype(np.uint8)
            im_target_rgb = cv2.resize(im_target_rgb, (600, 600))
            cv2.imshow("output", im_target_rgb)
            cv2.waitKey(10)

        # loss
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)

        loss.backward()
        optimizer.step()

        torch.save(model.state_dict(), 'results/model.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')

        print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())
        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break



# for batch_idx in range(args.maxIter):
#     # forwarding
#     optimizer.zero_grad()
#     output = model(data)[0]
#     output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
#
#
#     outputHP = output.reshape((im.shape[0], im.shape[1], args.nChannel))
#     HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
#     HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
#     lhpy = loss_hpy(HPy, HPy_target)
#     lhpz = loss_hpz(HPz, HPz_target)
#
#     ignore, target = torch.max(output, 1)
#     im_target = target.data.cpu().numpy()
#
#     # plt.imshow(im_target.reshape(191, 194))
#     # plt.show()
#
#     nLabels = len(np.unique(im_target))
#     if args.visualize:
#         im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
#         im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
#         im_target_rgb = cv2.resize(im_target_rgb, (600, 600))
#         data2 = cv2.resize(im, (600, 600))
#         cv2.imshow("output", im_target_rgb)
#         cv2.imshow("original", data2)
#         cv2.waitKey(10)
#
#     # loss
#     if args.scribble:
#         loss = args.stepsize_sim * loss_fn(output[inds_sim], target[inds_sim]) + args.stepsize_scr * loss_fn_scr(
#             output[inds_scr], target_scr[inds_scr]) + args.stepsize_con * (lhpy + lhpz)
#     else:
#         loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
#
#     loss.backward()
#     optimizer.step()
#
#     torch.save(model.state_dict(), 'results/model.pth')
#     torch.save(optimizer.state_dict(), 'results/optimizer.pth')
#
#     print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())
#
#     if nLabels <= args.minLabels:
#         print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
#         break
#
#
#
# # save output image
# if not args.visualize:
#     output = model(data)[0]
#     output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
#     ignore, target = torch.max(output, 1)
#     im_target = target.data.cpu().numpy()
#     im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
#     im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
#     cv2.imwrite("output.png", im_target_rgb)
