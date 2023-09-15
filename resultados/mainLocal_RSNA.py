# para fazer a gpu funcionar no pytorch
# siga este tutorial https://pub.towardsai.net/installing-pytorch-with-cuda-support-on-windows-10-a38b1134535e
# e finalmente instale pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html




# from __future__ import print_function
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.init
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nb
from dicom_to_nifti import converter
from datetime import datetime



use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=50, type=int, help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=5, type=int, help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, help='learning rate')
parser.add_argument('--nConv', metavar='M', default=3, type=int, help='number of convolutional layers')
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=1, type=float, help='step size for scribble loss')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, help='visualization flag')
# parser.add_argument('--input', metavar='FILENAME', default=r'D:\Users\paulo\PycharmProjects\pytorch-unsupervised-segmentation-tip\imagens\3.png', help='input image file name', required=False)
parser.add_argument('--input', metavar='FILENAME',
                    default=r'E:\PycharmProjects\pythonProject\imagens\Dsc32909.jpg',
                    help='input image file name', required=False)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float, help='step size for similarity loss', required=False)

args = parser.parse_args()
agora = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")



def criar_pasta(id_exame):
    pasta = "E:/PycharmProjects/pythonProject/results/" + agora + "/" + id_exame + "/"
    if not os.path.isdir(pasta):
        os.makedirs(pasta)
    return pasta


def plotarHistograma(exame):
    # plt.hist(exame.flatten(), bins=80, color='c')
    plt.hist(exame.flatten(), color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()



# E:\PycharmProjects\pythonProject\exame\CQ500CT234\Unknown Study\CT 55mm Plain
# folder_dcm = r"E:\PycharmProjects\pythonProject\exame\CQ500CT257\Unknown Study\CT 0.625mm"
folder_dcm = r"D:\dataset\rsna-miccai-brain-tumor-radiogenomic-classification\train\00002\T2w"
nifti_file = r"E:\PycharmProjects\pythonProject\build\rsna_miccai_T2w_00002.nii.gz"
pasta = criar_pasta("rsna_miccai_T2w_00002")





# folder_dcm = r"D:\dataset\rsna-miccai-brain-tumor-radiogenomic-classification\train\00002\T2w"
# nifti_file = r"E:\PycharmProjects\pythonProject\build\rsna_miccai_T2w_00002.nii.gz"



vol = converter(folder_dcm, nifti_file)



Z = vol.shape[0]
H = 512
W = 512
exame1 = vol.reshape(Z,512,512)



# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        '''

        :param input_dim:
        '''
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
# data_old = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))

# data = torch.from_numpy(np.array([exame1.astype('float32') / 255.]))
data = torch.from_numpy(np.array([exame1.astype('float32')]))
data = data.reshape(Z,512,512)

if use_cuda:
    data = data.cuda()
data = Variable(data)


# train
# model = MyNet(data.size(1))
model = MyNet(1)
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

# HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], args.nChannel)
# HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, args.nChannel)

HPy_target = torch.zeros(512 - 1, 512, args.nChannel)
HPz_target = torch.zeros(512, 512 - 1, args.nChannel)

if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))
# label_colours = np.random.randint(255, size=(100, 1))




parou = False

for batch_idx in range(args.maxIter):
    if parou:
        break
    for slice in range(Z)[52:320]:
        # data1 = exame1[slice, :, :] exame in natura
        data1 = data[slice, :, :]  #exame na escala de cinza 255
        # data = torch.from_numpy(data1.reshape(1,1,512,512).astype('float32'))
        data1 = data1.reshape(1,1,512,512)
        if use_cuda:
            data1 = data1.cuda()
        data1 = Variable(data1)


        # forwarding
        optimizer.zero_grad()
        output1 = model(data1)[0]



        # plt.imshow(output1[0,:,:].data.cpu().numpy())
        # plt.show()

        output = output1.permute(1, 2, 0).contiguous().view(-1, args.nChannel)


        posicao = 0
        rows, cols = 5, 5
        plt.figure(figsize=(60, 40))
        fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')

        for row in range(rows):
            for col in range(cols):
                # ax[row, col].text(0.5, 0.5,
                #                   str((row, col)),
                #                   color="green",
                #                   fontsize=18,
                #                   ha='center')
                ax[row, col].imshow(output1[posicao,:,:].data.cpu().numpy())
                posicao = posicao + 1

        plt.show()




        # plt.imshow(output.data.cpu().numpy())
        # plt.show()

        outputHP = output.reshape((data1.shape[2], data1.shape[3], args.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]

        # continuity loss definition
        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)


        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        data1 = target.data.cpu().numpy()
        data1 = data1.reshape(512,512).astype(np.uint8)
        # plt.imshow(im_target.reshape(191, 194))
        # plt.show()

        nLabels = len(np.unique(im_target))


        if args.visualize:
            im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(512,512,3).astype(np.uint8)

            # nifti_teste[slice, :, :] = im_target_rgb
            # nifti_teste[slice, :, :] = im_target.reshape(512,512).astype(np.uint8)


            im_target_rgb = cv2.resize(im_target_rgb, (1200, 1200))
            # Saving the image

            pastaFinal = pasta + "/" + str(nLabels) + "/"

            if not os.path.isdir(pastaFinal):
                os.makedirs(pastaFinal)


            cv2.imwrite(pastaFinal + str(slice) + ".jpg", im_target_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 400])

            data2 = cv2.resize(data1, (1200, 1200))
            cv2.imshow("output", im_target_rgb)
            cv2.imshow("original", data2)
            cv2.waitKey(10)

        # loss
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)

        loss.backward()
        optimizer.step()

        torch.save(model.state_dict(), r'E:/PycharmProjects/pythonProject/results/model.pth')
        torch.save(optimizer.state_dict(), r'E:/PycharmProjects/pythonProject/results/optimizer.pth')

        print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            parou = True
            break




model.train(False)

exames_para_teste = [r"E:\PycharmProjects\pythonProject\exame\CQ500CT342\Unknown Study\CT 0.625mm",
                     r"E:\PycharmProjects\pythonProject\exame\CQ500CT396\Unknown Study\CT 0.625mm"]

for folder_dcm_teste in exames_para_teste:
    # folder_dcm_teste = r"E:\PycharmProjects\pythonProject\exame\CQ500CT420\Unknown Study\CT 0.625mm"

    exame_id = folder_dcm_teste.split("\\")[4]
    nifti_file_teste = "E:\\PycharmProjects\\pythonProject\\exame\\" + exame_id + ".nii.gz"
    pasta_teste = criar_pasta(exame_id + "_validacao")

    exame_teste = converter(folder_dcm_teste, nifti_file_teste)


    fileName = exame_id + str(args.minLabels) + '.nii.gz'
    img = nb.Nifti1Image(exame_teste.T, np.eye(4))
    nb.save(img, os.path.join('E:/PycharmProjects/pythonProject/build', fileName))

    Z = exame_teste.shape[0]
    exame1_teste = exame_teste.reshape(Z,512,512)
    nifti_teste = np.ones((Z, 512, 512), dtype=np.uint8) # dummy data in numpy matrix

    for slice in range(Z):
        data1 = exame1_teste[slice, :, :]
        data_teste = torch.from_numpy(data1.reshape(1, 1, 512, 512).astype('float32'))
        if use_cuda:
            data_teste = data_teste.cuda()
        data_teste = Variable(data_teste)
        output_teste = model(data_teste)[0]
        output = output_teste.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()



        im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(512, 512, 3).astype(np.uint8)

        nifti_teste[slice, :, :] = im_target.reshape(512, 512).astype(np.uint8)

        im_target_rgb = cv2.resize(im_target_rgb, (1200, 1200))
        cv2.imwrite(pasta_teste + str(slice) + ".jpg", im_target_rgb , [int(cv2.IMWRITE_JPEG_QUALITY), 400])

        data2 = cv2.resize(data1, (1200, 1200))
        cv2.imshow("output", im_target_rgb)
        cv2.imshow("original", data2)
        cv2.waitKey(10)





    fileName = 'result_'+exame_id+'_'+ str(args.minLabels) + '.nii.gz'
    img = nb.Nifti1Image(nifti_teste.T, np.eye(4))  # Save axis for data (just identity)
    img.header.get_xyzt_units()
    img.to_filename(os.path.join('E:/PycharmProjects/pythonProject/build', fileName))  # Save as NiBabel file


