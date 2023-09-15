'''
responsavel por executar o teste com optuna
'''


import os
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
import numpy as np
# !pip install optuna
import optuna
import cufflinks
import plotly
import matplotlib as plot
import joblib



# Argumentos necessários para que a rede possa executar
### nChannel = 100 - Quantidade de neuronios usados nas camadas da deep learning de segmentação.
### maxIter = 50 - Quantidade de iterações de treinamento.
### minLabels = 3 - Quantidade mínima de segmentos/regiões que a rede deverá criar para cada imagem.
### lr = 0.1 - Taxa de aprendizado utilizado na deep learning.
### nConv = 3 - Quantidade de camadas convolucionais no terceiro bloco da deep learning.
### visualize - Responsável por exibir imagens durante a execução do treinamento
### stepsize_sim = 1 - Tamanho do passo na loss de similaridade
### stepsize_con = 1 - Tamanho do passo na loss de continuidade

# DEVICE = torch.device("cuda")  ##'cuda' or 'cpu'


use_cuda = torch.cuda.is_available()

def define_model(trial):

    layers = []
    nChannel = trial.suggest_int(name="nChannel", low=15, high=300, step=15)
    nConv = trial.suggest_int(name="nConv", low=1, high=9, step=2)

    # CNN model
    class MyNet(nn.Module):
        def __init__(self, input_dim):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(nChannel)
            self.conv2 = nn.ModuleList()
            self.bn2 = nn.ModuleList()
            for i in range(nConv - 1):
                self.conv2.append(nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
                self.bn2.append(nn.BatchNorm2d(nChannel))
            self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0)
            self.bn3 = nn.BatchNorm2d(nChannel)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn1(x)

            for i in range(nConv - 1):
                x = self.conv2[i](x)
                x = F.relu(x)
                x = self.bn2[i](x)
            x = self.conv3(x)
            x = self.bn3(x)
            return x

    model = MyNet(1)
    return model, nChannel

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

def getDataExame():
    folder_dcm = r"E:\PycharmProjects\pythonProject\exame\CQ500CT257\Unknown Study\CT 0.625mm"
    exame = np.array([read_dicom_file(folder_dcm)])
    exame1 = exame.reshape(256, 512, 512)
    return exame1


def objective(trial):


    # Generate the model.
    model, nChannel_2 = define_model(trial)
    print(model)
    if use_cuda:
        model.cuda()



    maxIter = trial.suggest_int(name="maxIter", low=5, high=15, step=5)
    # Generate the optimizers.
    stepsize_sim = trial.suggest_float(name="stepsize_sim", low=1, high=2, step=0.5)
    stepsize_con = trial.suggest_float(name="stepsize_con", low=1, high=2, step=0.5)
    minLabels = trial.suggest_int(name="minLabels", low=3, high=6, step=1)
    # nChannel_2 = trial.suggest_int(name="nChannel_2", low=15, high=150, step=15)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # for hp tuning
    # optimizer_name = "Adam"
    lr = trial.suggest_float(name="lr", low=0.001, high=0.1, log=True)
    # momentum = trial.suggest_float(name="momentum", low=0.9, high=0.99, log=True)
    # lr = 0.001
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, momentum=momentum)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)


    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()
    # scribble loss definition
    loss_fn_scr = torch.nn.CrossEntropyLoss()
    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(size_average=True)
    loss_hpz = torch.nn.L1Loss(size_average=True)

    # criando objetivos, ou seja, criando tensores compostos de zeros
    # esses tensores serão utilizados como objetivos para poder reduzir os valores de perda no treinamento
    # não supervisionado das redes neurais
    HPy_target = torch.zeros(512 - 1, 512, nChannel_2)
    HPz_target = torch.zeros(512, 512 - 1, nChannel_2)

    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

    # instanciando o algoritmo de gradiente descendente
    # com parâmetros de learning rate e momentum
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)



    # Get the dataset.

    exame1 = getDataExame()

    for batch_idx in range(maxIter):
        model.train()
        parou = False
        for slice in range(256):
            data1 = exame1[slice, :, :]
            data = torch.from_numpy(data1.reshape(1, 1, 512, 512).astype('float32'))
            if use_cuda:
                data = data.cuda()
            data = Variable(data)

            # forwarding
            optimizer.zero_grad()
            output1 = model(data)[0]
            output = output1.permute(1, 2, 0).contiguous().view(-1, nChannel_2)

            # plt.imshow(output.data.cpu().numpy())
            # plt.show()

            outputHP = output.reshape((data.shape[2], data.shape[3], nChannel_2))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)

            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()

            # plt.imshow(im_target.reshape(191, 194))
            # plt.show()

            nLabels = len(np.unique(im_target))

            loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)

            loss.backward()
            optimizer.step()

            # torch.save(model.state_dict(), 'results/model.pth')
            # torch.save(optimizer.state_dict(), 'results/optimizer.pth')

            #             print(batch_idx, '/', maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item(), ' | slice :', slice)
            if nLabels <= minLabels:
                #                 trial.report(batch_idx, loss)
                #                 trial.report(batch_idx, nLabels)
                print("nLabels", nLabels, "reached minLabels", minLabels, ".")
                parou = True
                break

        if parou:
            break
        trial.report(loss, nLabels)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss  # accuracy


if __name__ == "__main__":
    #study = optuna.create_study(direction="maximize")  # 'maximize' because objective function is returning accuracy
    study = optuna.create_study(direction="minimize")  # 'minimize' because objective function is returning loss
    study.optimize(objective, n_trials=30)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    study.best_trial

    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_intermediate_values(study).show()

    optuna.visualization.plot_contour(study).show()

    optuna.visualization.plot_edf(study).show()

    optuna.visualization.plot_param_importances(study).show()  ## this is important to figure out which hp is important

    optuna.visualization.plot_slice(study) .show() ## this gives a clear picture

    optuna.visualization.plot_parallel_coordinate(study).show()

    savepath = 'C:\\Users\\paulo\\PycharmProjects\\optuna_pytorch\optuna\\'
    joblib.dump(study, f"{savepath}xgb_optuna_study_batch.pkl")
    jl = joblib.load(f"{savepath}xgb_optuna_study_batch.pkl")
    print(jl.best_trial.params)

    a = 45