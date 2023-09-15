# para fazer a gpu funcionar no pytorch
# siga este tutorial https://pub.towardsai.net/installing-pytorch-with-cuda-support-on-windows-10-a38b1134535e
# e finalmente instale pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html




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

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=50, type=int, help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, help='learning rate')
parser.add_argument('--nConv', metavar='M', default=3, type=int, help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, help='visualization flag')
# parser.add_argument('--input', metavar='FILENAME', default=r'D:\Users\paulo\PycharmProjects\pytorch-unsupervised-segmentation-tip\imagens\3.png', help='input image file name', required=False)
parser.add_argument('--input', metavar='FILENAME',
                    default=r'E:\PycharmProjects\pythonProject\imagens\Dsc32909.jpg',
                    help='input image file name', required=False)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float, help='step size for similarity loss',
                    required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, help='step size for scribble loss')
args = parser.parse_args()

# E:\PycharmProjects\pythonProject\imagens\Dsc32907+.jpg
# E:\PycharmProjects\pythonProject\imagens\Dsc32909.jpg

# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv - 1):
            self.conv2.append(nn.Conv3d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm3d(args.nChannel))
        self.conv3 = nn.Conv3d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(args.nChannel)

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


# load 3d image like nift or dicom
im = cv2.imread(args.input)
data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))



if use_cuda:
    data = data.cuda()
data = Variable(data)



# train
model = MyNet(data.size(1))
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

HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], args.nChannel)
HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, args.nChannel)
if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))

for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

    # plt.imshow(output.data.cpu().numpy())
    # plt.show()

    outputHP = output.reshape((im.shape[0], im.shape[1], args.nChannel))
    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy, HPy_target)
    lhpz = loss_hpz(HPz, HPz_target)

    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()

    # plt.imshow(im_target.reshape(191, 194))
    # plt.show()

    nLabels = len(np.unique(im_target))
    if args.visualize:
        im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        im_target_rgb = cv2.resize(im_target_rgb, (600, 600))
        data2 = cv2.resize(im, (600, 600))
        cv2.imshow("output", im_target_rgb)
        cv2.imshow("original", data2)
        cv2.waitKey(10)

    # loss
    if args.scribble:
        loss = args.stepsize_sim * loss_fn(output[inds_sim], target[inds_sim]) + args.stepsize_scr * loss_fn_scr(
            output[inds_scr], target_scr[inds_scr]) + args.stepsize_con * (lhpy + lhpz)
    else:
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)

    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), 'results/model.pth')
    torch.save(optimizer.state_dict(), 'results/optimizer.pth')

    print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

    if nLabels <= args.minLabels:
        print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# teste the network

teste_image_url = r'D:\Users\paulo\PycharmProjects\pytorch-unsupervised-segmentation-tip\imagens\Normal-74-yo-CT-head-5_pt.jpg'
im_teste = cv2.imread(teste_image_url)
data_teste = torch.from_numpy(np.array([im_teste.transpose((2, 0, 1)).astype('float32') / 255.]))

if use_cuda:
    data_teste = data_teste.cuda()
data_teste = Variable(data_teste)

if use_cuda:
    model.cuda()
model.eval()


test_loss = 0
correct = 0
output = model(data_teste)

output = output.reshape((args.nChannel, im_teste.shape[1], im_teste.shape[0] ))

output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

outputHP = output.reshape((im_teste.shape[0], im_teste.shape[1], args.nChannel))
HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
# lhpy = loss_hpy(HPy, HPy_target)
# lhpz = loss_hpz(HPz, HPz_target)

ignore, target = torch.max(output, 1)
im_target = target.data.cpu().numpy()

# plt.imshow(im_target.reshape(191, 194))
# plt.show()

nLabels = len(np.unique(im_target))
if args.visualize:
    im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im_teste.shape).astype(np.uint8)
    cv2.imshow("output_teste", im_target_rgb)
    cv2.waitKey(10)

# save output image
if not args.visualize:
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im_teste.shape).astype(np.uint8)
    cv2.imwrite("output.png", im_target_rgb)
