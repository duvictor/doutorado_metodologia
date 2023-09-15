'''
responsável por realizar uma busca de melhores parâmetros

a ideia é ter uma fórmula matemática para explicar os melhores parâmetros
'''

# para fazer a gpu funcionar no pytorch
# siga este tutorial https://pub.towardsai.net/installing-pytorch-with-cuda-support-on-windows-10-a38b1134535e
# e finalmente instale pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html




# from __future__ import print_function
import argparse
import os
from functools import partial
import pydicom
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
import matplotlib.pyplot as plt
import random
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nb


import ray
ray.init()


from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



use_cuda = torch.cuda.is_available()

# parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
#
# parser.add_argument('--nChannel', metavar='N', default=100, type=int, help='number of channels')
# parser.add_argument('--maxIter', metavar='T', default=50, type=int, help='number of maximum iterations')
# parser.add_argument('--minLabels', metavar='minL', default=5, type=int, help='minimum number of labels')
# parser.add_argument('--lr', metavar='LR', default=0.1, type=float, help='learning rate')
# parser.add_argument('--nConv', metavar='M', default=3, type=int, help='number of convolutional layers')
# parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, help='visualization flag')
# # parser.add_argument('--input', metavar='FILENAME', default=r'D:\Users\paulo\PycharmProjects\pytorch-unsupervised-segmentation-tip\imagens\3.png', help='input image file name', required=False)
# parser.add_argument('--input', metavar='FILENAME',
#                     default=r'E:\PycharmProjects\pythonProject\imagens\Dsc32909.jpg',
#                     help='input image file name', required=False)
# parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float, help='step size for similarity loss',
#                     required=False)
# parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, help='step size for continuity loss')
# parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, help='step size for scribble loss')
# args = parser.parse_args()






def get_pixels_hu_2(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # realiza o filtro de ar
    # Ar	−1000
    # Pulmão	−500
    # Gordura	−100 a −50
    # Água	0
    # Fluido cerebroespinhal	15
    # Rim	30
    # Sangue	+30 a +45
    # Músculo	+10 a +40
    # Massa cinzenta	+37 a +45
    # Massa branca	+20 a +30
    # Fígado	+40 a +60
    # Tecidos moles, Contraste	+100 a +300
    # Osso	+700 (osso esponjoso) a +3000 (osso denso)
    image[image <= -2000] = 0
    image_original = image
    plt.imshow(image_original[0, :, ])
    plt.show()

    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):



        if slope != 1:
            image_original[slice_number] = slope * image_original[slice_number].astype(np.float64)
            image_original[slice_number] = image_original[slice_number].astype(np.int16)

        image_original[slice_number] += np.int16(intercept)

    return np.array(image_original, dtype=np.int16)


def resample3d(image):
    # Determine current pixel spacing
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128

    # Get current depth

    current_depth = image.shape[0]
    current_width = image.shape[1]
    current_height = image.shape[2]

    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    image = scipy.ndimage.interpolation.zoom(image, (depth_factor, width_factor, height_factor), order=1)
    image = np.transpose(image)
    return image


def read_dicom_file(filepath):
    slices = [pydicom.read_file(filepath + '/' + s) for s in os.listdir(filepath)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    patient1_hu_scans = get_pixels_hu_2(slices)

    # images = resample3d(patient1_hu_scans)
    #plot_3d(images)
    # return images

    return patient1_hu_scans


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    p = p[:, :, ::-1]

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def plotarHistograma(exame):
    # plt.hist(exame.flatten(), bins=80, color='c')
    plt.hist(exame.flatten(), color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()




def train(config, checkpoint_dir=None, data_dir=None):
    '''
    # config['nChannel']
    # config['nConv']
    # config['lr']
    # config['momentum'] = 0.9
    # config['maxIter']
    # config['stepsize_sim']
    # config['stepsize_con']
    # config['minLabels']

    :param config:
    :param checkpoint_dir:
    :param data_dir:
    :return:
    '''



    folder_dcm = r"E:\PycharmProjects\pythonProject\exame\CQ500CT257\Unknown Study\CT 0.625mm"

    # files_dcm = [os.path.join(os.getcwd(), folder_dcm, x) for x in os.listdir(folder_dcm)]
    # exame = np.array([read_dicom_file(path) for path in files_dcm])
    exame = np.array([read_dicom_file(folder_dcm)])
    exame1 = exame.reshape(256,512,512)

    plotarHistograma(exame1)

    # Our plot function takes a threshold argument which we can use to plot certain structures, such as all tissue or only the bones.
    # 400 is a good threshold for showing the bones only (see Hounsfield unit table above). Let's do this!
    # plot_3d(exame1, 400)


    # CNN model
    class MyNet(nn.Module):
        def __init__(self, input_dim):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(input_dim, config['nChannel'], kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(config['nChannel'])
            self.conv2 = nn.ModuleList()
            self.bn2 = nn.ModuleList()
            for i in range(config['nConv'] - 1):
                self.conv2.append(nn.Conv2d(config['nChannel'], config['nChannel'], kernel_size=3, stride=1, padding=1))
                self.bn2.append(nn.BatchNorm2d(config['nChannel']))
            self.conv3 = nn.Conv2d(config['nChannel'], config['nChannel'], kernel_size=1, stride=1, padding=0)
            self.bn3 = nn.BatchNorm2d(config['nChannel'])

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn1(x)

            for i in range(config['nConv'] - 1):
                x = self.conv2[i](x)
                x = F.relu(x)
                x = self.bn2[i](x)
            x = self.conv3(x)
            x = self.bn3(x)
            return x




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

    HPy_target = torch.zeros(512 - 1, 512, config['nChannel'])
    HPz_target = torch.zeros(512, 512 - 1, config['nChannel'])

    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    label_colours = np.random.randint(255, size=(100, 3))
    # label_colours = np.random.randint(255, size=(100, 1))

    nifti_teste = np.ones((256, 512, 512, 3), dtype=np.uint8) # dummy data in numpy matrix



    for batch_idx in range(config['maxIter']):
        for slice in range(256):
            data1 = exame1[slice, :, :]
            data = torch.from_numpy(data1.reshape(1,1,512,512).astype('float32'))
            if use_cuda:
                data = data.cuda()
            data = Variable(data)


            # forwarding
            optimizer.zero_grad()
            output1 = model(data)[0]
            output = output1.permute(1, 2, 0).contiguous().view(-1, config['nChannel'])

            # plt.imshow(output.data.cpu().numpy())
            # plt.show()

            outputHP = output.reshape((data.shape[2], data.shape[3], config['nChannel']))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)

            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()

            # plt.imshow(im_target.reshape(191, 194))
            # plt.show()

            nLabels = len(np.unique(im_target))


            # if args.visualize:
            im_target_rgb = np.array([label_colours[c % config['nChannel']] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(512,512,3).astype(np.uint8)

            nifti_teste[slice, :, :] = im_target_rgb


            im_target_rgb = cv2.resize(im_target_rgb, (600, 600))
            data2 = cv2.resize(data1, (600, 600))
            cv2.imshow("output", im_target_rgb)
            cv2.imshow("original", data2)
            cv2.waitKey(10)

            # loss
            loss = config['stepsize_sim'] * loss_fn(output, target) + config['stepsize_con'] * (lhpy + lhpz)

            loss.backward()
            optimizer.step()

            torch.save(model.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')

            print(batch_idx, '/', config['maxIter'], '|', ' label num :', nLabels, ' | loss :', loss.item())
            # tune.report(batch_idx=batch_idx, maxIter=config['maxIter'], nLabels=nLabels, loss=(loss), stepsize_sim=config['stepsize_sim'], stepsize_con=config['stepsize_con'], nChannel=config['nChannel'])

            if nLabels <= config['minLabels']:
                print("nLabels", nLabels, "reached minLabels", config['minLabels'], ".")
                break

    tune.report(nChannel=config['nChannel'], nConv=config['nConv'], lr=config['lr'], momentum=config['momentum'], maxIter=config['maxIter'], stepsize_sim= config['stepsize_sim'], stepsize_con=config['stepsize_con'], minLabels=config['minLabels'])
    print("Finished Training")
 # config['nChannel']
# config['nConv']
# config['lr']
# config['momentum'] = 0.9
# config['maxIter']
# config['stepsize_sim']
# config['stepsize_con']
# config['minLabels']



def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):

    config = {
        "nChannel": tune.choice([10, 30, 50, 10, 150]),
        "nConv": tune.choice([1, 3, 5, 7]),
        "lr": tune.choice([0.1, 0.01, 0.001, 0.0001]),
        "momentum": tune.choice([0.9, 0.99, 0.999]),
        "maxIter": tune.choice([5, 10, 15]),
        "stepsize_sim": tune.choice([0.5, 1, 1.5]),
        "stepsize_con": tune.choice([0.5, 1, 1.5]),
        "minLabels": tune.choice([6, 5, 4, 3])
    }

    # numero maximo de iterações
    max_num_epochs = 15
    num_samples = 10
    data_dir = os.path.abspath("./data")



    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["nChannel", "nConv", "lr", "momentum", "maxIter", "stepsize_sim", "stepsize_con", "minLabels"])

    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True)

    a = 45
    b = 90

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)