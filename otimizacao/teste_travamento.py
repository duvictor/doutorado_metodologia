import joblib
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
# from torchsummary import summary
import numpy as np
# !pip install optuna
import optuna
import cufflinks
import plotly
import matplotlib as plot
import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
savepath = r'C:\Users\paulo\PycharmProjects\doutorado_metodologia\optuna\\'
jl = joblib.load(f"{savepath}xgb_optuna_study_batch_6.pkl")

figure1 = optuna.visualization.plot_param_importances(jl)
figure1.write_image('plot_param_importances_optuna_hyperparameter_find_label6.svg')

figure2 = optuna.visualization.plot_optimization_history(jl)
figure2.write_image('plot_optimization_history_optuna_hyperparameter_find_label6.svg')

figure3 = optuna.visualization.plot_parallel_coordinate(jl)
figure3.write_image('plot_parallel_coordinate_optuna_hyperparameter_find_label6.svg')

figure4 = optuna.visualization.plot_contour(jl)
figure4.write_image('plot_contour_optuna_hyperparameter_find_label6.svg')

figure5 = optuna.visualization.plot_intermediate_values(jl)
figure5.write_image('plot_intermediate_values_optuna_hyperparameter_find_label6.svg')

figure6 = optuna.visualization.plot_slice(jl)
figure6.write_image('plot_slice_optuna_hyperparameter_find_label6.svg')

figure7 = optuna.visualization.plot_edf(jl)
figure7.write_image('plot_edf_optuna_hyperparameter_find_label6.svg')
