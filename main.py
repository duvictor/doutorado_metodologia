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
from validarMetodologia import executar_metodologia



use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=50, type=int, help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=6, type=int, help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, help='learning rate')
parser.add_argument('--nConv', metavar='M', default=3, type=int, help='number of convolutional layers')
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, help='step size for continuity loss - regularização')
parser.add_argument('--stepsize_scr', metavar='SCR', default=1, type=float, help='step size for scribble loss')
parser.add_argument('--similar', metavar='sim', default=1, type=float, help='medir a distância entre uma imagem e outra na questão da similaridade')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, help='visualization flag')
# parser.add_argument('--input', metavar='FILENAME', default=r'D:\Users\paulo\PycharmProjects\pytorch-unsupervised-segmentation-tip\imagens\3.png', help='input image file name', required=False)
parser.add_argument('--input', metavar='FILENAME',
                    default=r'E:\PycharmProjects\pythonProject\imagens\Dsc32909.jpg',
                    help='input image file name', required=False)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float, help='step size for similarity loss - regularizacao', required=False)

args = parser.parse_args()

# E:\PycharmProjects\pythonProject\imagens\Dsc32907+.jpg
# E:\PycharmProjects\pythonProject\imagens\Dsc32909.jpg




# def get_pixels_hu_2(scans):
#     image = np.stack([s.pixel_array for s in scans])
#     # Convert to int16 (from sometimes int16),
#     # should be possible as values should always be low enough (<32k)
#     image = image.astype(np.int16)
#
#     # Set outside-of-scan pixels to 0
#     # The intercept is usually -1024, so air is approximately 0
#     # image[image == -2000] = 0
#
#     # Convert to Hounsfield units (HU)
#     intercept = scans[0].RescaleIntercept
#     slope = scans[0].RescaleSlope
#
#     if slope != 1:
#         image = slope * image.astype(np.float64)
#         image = image.astype(np.int16)
#
#     image += np.int16(intercept)
#
#     return np.array(image, dtype=np.int16)
#
#
# def get_pixels_hu(slices):
#     image = np.stack([s.pixel_array for s in slices])
#     # Convert to int16 (from sometimes int16),
#     # should be possible as values should always be low enough (<32k)
#     image = image.astype(np.int16)
#
#     # Set outside-of-scan pixels to 0
#     # The intercept is usually -1024, so air is approximately 0
#     # realiza o filtro de ar
#     # Ar	−1000
#     # Pulmão	−500
#     # Gordura	−100 a −50
#     # Água	0
#     # Fluido cerebroespinhal	15
#     # Rim	30
#     # Sangue	+30 a +45
#     # Músculo	+10 a +40
#     # Massa cinzenta	+37 a +45
#     # Massa branca	+20 a +30
#     # Fígado	+40 a +60
#     # Tecidos moles, Contraste	+100 a +300
#     # Osso	+700 (osso esponjoso) a +3000 (osso denso)
#     # image[image <= -2000] = 0
#     image_original = image
#     plt.imshow(image_original[0, :, ])
#     plt.show()
#
#     intercept = slices[0].RescaleIntercept
#     slope = slices[0].RescaleSlope
#
#     # Convert to Hounsfield units (HU)
#     for slice_number in range(len(slices)):
#
#
#
#         if slope != 1:
#             image_original[slice_number] = slope * image_original[slice_number].astype(np.float64)
#             image_original[slice_number] = image_original[slice_number].astype(np.int16)
#
#         image_original[slice_number] += np.int16(intercept)
#
#     return np.array(image_original, dtype=np.int16)
#
#
# def resample3d(image):
#     # Determine current pixel spacing
#     # Set the desired depth
#     desired_depth = 64
#     desired_width = 128
#     desired_height = 128
#
#     # Get current depth
#
#     current_depth = image.shape[0]
#     current_width = image.shape[1]
#     current_height = image.shape[2]
#
#     # Compute depth factor
#     depth = current_depth / desired_depth
#     width = current_width / desired_width
#     height = current_height / desired_height
#     depth_factor = 1 / depth
#     width_factor = 1 / width
#     height_factor = 1 / height
#
#     image = scipy.ndimage.interpolation.zoom(image, (depth_factor, width_factor, height_factor), order=1)
#     image = np.transpose(image)
#     return image
#

# def read_dicom_file(filepath):
#     slices = [pydicom.read_file(filepath + '/' + s) for s in os.listdir(filepath)]
#     slices.sort(key=lambda x: int(x.InstanceNumber))
#     try:
#         slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
#     except:
#         slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
#
#     for s in slices:
#         s.SliceThickness = slice_thickness
#     # patient1_hu_scans = get_pixels_hu_2(slices)
#
#     vol, pixdim, mat  = get_pixels_from_dicom(slices)
#     # patient1_hu_scans = slices
#     # images = resample3d(patient1_hu_scans)
#     #plot_3d(images)
#     # return images
#
#     return vol, pixdim, mat


# def plot_3d(image, threshold=-300):
# #     # Position the scan upright,
# #     # so the head of the patient would be at the top facing the camera
# #     p = image.transpose(2, 1, 0)
# #     p = p[:, :, ::-1]
# #
# #     verts, faces = measure.marching_cubes(p, threshold)
# #
# #     fig = plt.figure(figsize=(10, 10))
# #     ax = fig.add_subplot(111, projection='3d')
# #
# #     # Fancy indexing: `verts[faces]` to generate a collection of triangles
# #     mesh = Poly3DCollection(verts[faces], alpha=0.1)
# #     face_color = [0.5, 0.5, 1]
# #     mesh.set_facecolor(face_color)
# #     ax.add_collection3d(mesh)
# #
# #     ax.set_xlim(0, p.shape[0])
# #     ax.set_ylim(0, p.shape[1])
# #     ax.set_zlim(0, p.shape[2])
# #
# #     plt.show()








def plotarHistograma(exame):
    # plt.hist(exame.flatten(), bins=80, color='c')
    plt.hist(exame.flatten(), color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

# folder_dcm = r"E:\PycharmProjects\pythonProject\exame\CQ500CT257\Unknown Study\CT 0.625mm"
folder_dcm = r"D:\dataset_cq500\CQ500CT47 CQ500CT47\Unknown Study\CT PRE CONTRAST THIN"
nifti_file = r"E:\PycharmProjects\pythonProject\build\CQ500CT47.nii.gz"

# folder_dcm = r"D:\dataset\rsna-miccai-brain-tumor-radiogenomic-classification\train\00002\T2w"
# nifti_file = r"E:\PycharmProjects\pythonProject\build\rsna_miccai_T2w_00002.nii.gz"

# files_dcm = [os.path.join(os.getcwd(), folder_dcm, x) for x in os.listdir(folder_dcm)]
# exame = np.array([read_dicom_file(path) for path in files_dcm])

# vol, pixdim, mat  = read_dicom_file(folder_dcm)

# convert DICOM coords to NIFTI coords (in-place)

# window_center = 40
# window_width = 80


window_center = 40
window_width = 40

janelamento = True
vol, affine = converter(folder_dcm, nifti_file, window_center, window_width)




# exame = np.array([read_dicom_file(folder_dcm)])
#remove the shape (1,256,512,512) to (256,512,512)
# Z = 256
Z = vol.shape[0]
H = 512
W = 512
exame1 = vol.reshape(Z,512,512)

# aki,, salvar exame para verificar o que acontece com o reshape



# fileName = 'CQ500CT257_'+ str(args.minLabels) + '.nii.gz'
# img = nb.Nifti1Image(vol.T, np.eye(4))
# nb.save(img, os.path.join('build', fileName))




# plotarHistograma(exame1)

# Our plot function takes a threshold argument which we can use to plot certain structures, such as all tissue or only the bones.
# 400 is a good threshold for showing the bones only (see Hounsfield unit table above). Let's do this!
# plot_3d(exame1, 400)



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

data = torch.from_numpy(np.array([exame1.astype('float32') / 255.]))
data = data.reshape(Z,512,512)

if use_cuda:
    data = data.cuda()
data = Variable(data)

# load scribble
if args.scribble:
    mask = cv2.imread(args.input.replace('.' + args.input.split('.')[-1], '_scribble.png'), -1)
    mask = mask.reshape(-1)
    mask_inds = np.unique(mask)
    mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
    inds_sim = torch.from_numpy(np.where(mask == 255)[0])
    inds_scr = torch.from_numpy(np.where(mask != 255)[0])
    target_scr = torch.from_numpy(mask.astype(np.int))
    if use_cuda:
        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()
    target_scr = Variable(target_scr)
    # set minLabels
    args.minLabels = len(mask_inds)

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


HPy_target = torch.zeros(512 - args.similar, 512, args.nChannel)
HPz_target = torch.zeros(512, 512 - args.similar, args.nChannel)

if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))
# label_colours = np.random.randint(255, size=(100, 1))




parou = False

# iteração para finalizar o algoritmo em caso de não encontrar os rótulos
for batch_idx in range(args.maxIter):
    loss_medio = 0
    if parou:
        break
    for slice in range(Z):
        # data1 = exame1[slice, :, :] exame in natura
        data1 = data[slice, :, :]  #exame na escala de cinza 255
        # data = torch.from_numpy(data1.reshape(1,1,512,512).astype('float32'))
        data1 = data1.reshape(1,1,512,512)
        if use_cuda:
            data1 = data1.cuda()
        data1 = Variable(data1)


        # forwarding
        optimizer.zero_grad()
        # extrator de características
        caracteristicas = model(data1)[0]



        # plt.imshow(output1[0,:,:].data.cpu().numpy())
        # plt.show()
        # permutador 1
        permutado = caracteristicas.permute(1, 2, 0).contiguous().view(-1, args.nChannel)


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
                ax[row, col].imshow(caracteristicas[posicao,:,:].data.cpu().numpy())
                posicao = posicao + 1

        plt.show()




        # plt.imshow(output.data.cpu().numpy())
        # plt.show()
        # remodelador1
        outputHP = permutado.reshape((data1.shape[2], data1.shape[3], args.nChannel))

        # início cálculo continuidade espacial
        HPy = outputHP[args.similar:, :, :] - outputHP[0:-args.similar, :, :]
        HPz = outputHP[:, args.similar:, :] - outputHP[:, 0:-args.similar, :]

        # continuity loss definition
        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)


        ignore, target = torch.max(permutado, 1)
        im_target = target.data.cpu().numpy()
        data_exibicao = target.data.cpu().numpy()
        data_exibicao = data_exibicao.reshape(512,512).astype(np.uint8)
        # plt.imshow(im_target.reshape(191, 194))
        # plt.show()

        nLabels = len(np.unique(im_target))


        if args.visualize:
            im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(512,512,3).astype(np.uint8)

            # nifti_teste[slice, :, :] = im_target_rgb
            # nifti_teste[slice, :, :] = im_target.reshape(512,512).astype(np.uint8)


            im_target_rgb = cv2.resize(im_target_rgb, (600, 600))
            data2 = cv2.resize(data_exibicao, (600, 600))
            cv2.imshow("output", im_target_rgb)
            cv2.imshow("original", data2)
            cv2.waitKey(10)

        # loss
        # if args.scribble:
            # a = 45
            # loss = args.stepsize_sim * loss_fn(output[inds_sim], target[inds_sim]) + args.stepsize_scr * loss_fn_scr(
            # output[inds_scr], target_scr[inds_scr]) + args.stepsize_con * (lhpy + lhpz)
        # else:
            # perda recebe 1 * entropia cruzada + 1 * (mae de y + mae de z)
            #         1              * entropia cruzada(output, target) + 1 (
            # loss = args.stepsize_sim * loss_fn(output, target)
            loss = args.stepsize_sim * loss_fn(permutado, target) + args.stepsize_con * (lhpy + lhpz)

        loss.backward()
        optimizer.step()
        loss_medio += loss.item()

        torch.save(model.state_dict(), 'results/model.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')

        print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

        if nLabels <= args.minLabels:
            print("loss_final :", loss.item(), " loss_medio :", loss_medio / Z, " nLabels :", nLabels, " reached minLabels :", args.minLabels, ".")
            parou = True
            break



# # teste the network into the train exame
# # nifti_teste = np.ones((Z, 512, 512, 3), dtype=np.uint8) # dummy data in numpy matrix
# nifti_teste = np.ones((Z, 512, 512), dtype=np.uint8)
#
#
#
#
#
# fileName = 'train_segmentation_'+ str(args.minLabels) + '.nii.gz'
# img = nb.Nifti1Image(nifti_teste.T, np.eye(4))  # Save axis for data (just identity)
# img.header.get_xyzt_units()
# img.to_filename(os.path.join('build',fileName))  # Save as NiBabel file
#
# # fim teste the network into the train exame



# INICIO VALIDAÇÃO
exames_validar = {
"CQ500CT42": r"D:\dataset_cq500\CQ500CT42 CQ500CT42\Unknown Study\CT PRE CONTRAST THIN",
"CQ500CT195": r"D:\dataset_cq500\CQ500CT195 CQ500CT195\Unknown Study\CT PRE CONTRAST THIN",
"CQ500CT200": r"D:\dataset_cq500\CQ500CT200 CQ500CT200\Unknown Study\CT Thin Plain",
"CQ500CT299": r"D:\dataset_cq500\CQ500CT299 CQ500CT299\Unknown Study\CT Thin Plain",
"CQ500CT418": r"D:\dataset_cq500\CQ500CT418 CQ500CT418\Unknown Study\CT Thin Plain",
"CQ500CT422": r"D:\dataset_cq500\CQ500CT422 CQ500CT422\Unknown Study\CT PLAIN THIN",
"CQ500CT440": r"D:\dataset_cq500\CQ500CT440 CQ500CT440\Unknown Study\CT Thin Plain",
}




for key in exames_validar:
    exame_validar_local = exames_validar[key]
    loss_medio = executar_metodologia(use_cuda, model, label_colours, args, key, exame_validar_local, window_center, window_width)
    print("finalizou o exame {} com loss médio de {}".format(key, loss_medio))




# folder_dcm = r"E:\PycharmProjects\pythonProject\exame\CQ500CT102\Unknown Study\CT 0.625mm"
# folder_dcm = r"D:\dataset_cq500\CQ500CT195 CQ500CT195\Unknown Study\CT PRE CONTRAST THIN"
# nifti_file = r"E:\PycharmProjects\pythonProject\exame\CQ500CT195.nii.gz"
#
# exame_teste, affine = converter(folder_dcm, nifti_file, 40, 80)
# empty_header = nb.Nifti1Header()
# empty_header.get_data_shape()
#
# Z = exame_teste.shape[0]
#
# fileName = 'CQ500CT195_'+ str(args.minLabels) + '.nii.gz'
# img = nb.Nifti1Image(exame_teste.T, affine, empty_header)
# nb.save(img, os.path.join('build', fileName))
#
#
#
# # files_dcm = [os.path.join(os.getcwd(), folder_dcm, x) for x in os.listdir(folder_dcm)]
# # exame = np.array([read_dicom_file(path) for path in files_dcm])
# # exame_teste = np.array([read_dicom_file(folder_dcm)])
# exame1_teste = exame_teste.reshape(Z,512,512)
# nifti_teste = np.ones((Z, 512, 512), dtype=np.uint8) # dummy data in numpy matrix
#
# for slice in range(Z):
#     data1 = exame1_teste[slice, :, :]
#     data_teste = torch.from_numpy(data1.reshape(1, 1, 512, 512).astype('float32'))
#     if use_cuda:
#         data_teste = data_teste.cuda()
#     data_teste = Variable(data_teste)
#     output_teste = model(data_teste)[0]
#     output = output_teste.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
#     ignore, target = torch.max(output, 1)
#     im_target = target.data.cpu().numpy()
#
#
#
#     im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
#     im_target_rgb = im_target_rgb.reshape(512, 512, 3).astype(np.uint8)
#
#     nifti_teste[slice, :, :] = im_target.reshape(512, 512).astype(np.uint8)
#
#     im_target_rgb = cv2.resize(im_target_rgb, (600, 600))
#     data2 = cv2.resize(data1, (600, 600))
#     cv2.imshow("output", im_target_rgb)
#     cv2.imshow("original", data2)
#     cv2.waitKey(10)
#
#
#
#
#
# fileName = 'result_CQ500CT195_'+ str(args.minLabels) + '.nii.gz'
# img = nb.Nifti1Image(nifti_teste.T, affine, empty_header)  # Save axis for data (just identity)
# img.header.get_xyzt_units()
# img.to_filename(os.path.join('build',fileName))  # Save as NiBabel file




#
# teste_image_url = r'D:\Users\paulo\PycharmProjects\pytorch-unsupervised-segmentation-tip\imagens\Normal-74-yo-CT-head-5_pt.jpg'
# im_teste = cv2.imread(teste_image_url)
# data_teste = torch.from_numpy(np.array([im_teste.transpose((2, 0, 1)).astype('float32') / 255.]))
#
# if use_cuda:
#     data_teste = data_teste.cuda()
# data_teste = Variable(data_teste)
#
# if use_cuda:
#     model.cuda()
# model.eval()
#
#
# test_loss = 0
# correct = 0
# output = model(data_teste)
#
# output = output.reshape((args.nChannel, im_teste.shape[1], im_teste.shape[0] ))
#
# output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
#
# outputHP = output.reshape((im_teste.shape[0], im_teste.shape[1], args.nChannel))
# HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
# HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
# lhpy = loss_hpy(HPy, HPy_target)
# lhpz = loss_hpz(HPz, HPz_target)
#
# ignore, target = torch.max(output, 1)
# im_target = target.data.cpu().numpy()
#
# # plt.imshow(im_target.reshape(191, 194))
# # plt.show()
#
# nLabels = len(np.unique(im_target))
# if args.visualize:
#     im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
#     im_target_rgb = im_target_rgb.reshape(im_teste.shape).astype(np.uint8)
#     cv2.imshow("output_teste", im_target_rgb)
#     cv2.waitKey(10)
#
# # save output image
# if not args.visualize:
#     output = model(data)[0]
#     output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
#     ignore, target = torch.max(output, 1)
#     im_target = target.data.cpu().numpy()
#     im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
#     im_target_rgb = im_target_rgb.reshape(im_teste.shape).astype(np.uint8)
#     cv2.imwrite("output.png", im_target_rgb)
