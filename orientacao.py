'''
responsavel por verificar a orientacao dos exames e tentar corrigir as imagens em orientações erradas

https://www.kaggle.com/code/davidbroberts/determining-dicom-image-order
'''

import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt


# Get a list of images in a series
def get_series_list(directory, study, series):
    series_list = []
    for subdirs, dirs, files in os.walk(directory + '/' + study + "/" + series):
        series_list = os.listdir(directory + '/' + study + '/' + series)
    return series_list



# Convert the Image Orientation Patient tag cosine values into a text string of the plane.
# This represents the plane the image is 'closest to' .. it does not explain any obliqueness
def get_image_plane(loc):

    row_x = round(loc[0])
    row_y = round(loc[1])
    row_z = round(loc[2])
    col_x = round(loc[3])
    col_y = round(loc[4])
    col_z = round(loc[5])
    if (row_x, row_y, col_x, col_y) == (1,0,0,0):
        return "Coronal"
    if (row_x, row_y, col_x, col_y) == (0,1,0,0):
        return "Sagittal"
    if (row_x, row_y, col_x, col_y) == (1,0,0,1):
        return "Axial"
    return "Unknown"


# folder_dcm = r"E:\PycharmProjects\pythonProject\exame\CQ500CT420\Unknown Study\CT 0.625mm"
directory = 'E:\PycharmProjects\pythonProject\exame\CQ500CT420'
study = 'Unknown Study'
series = 'CT 0.625mm'
files = []

# Get a list of images for this study/series
series_list = get_series_list(directory, study, series)

if len(series_list) > 0:
    for f in series_list:
        # Read the image and get it's orientation and position tags
        image = pydicom.dcmread(f'{directory}/{study}/{series}/{f}')
        plane = get_image_plane(image[0x0020, 0x0037])

        # Make a list
        files.append([f, plane, float(image[0x0020, 0x0032].value[0]), float(image[0x0020, 0x0032].value[1]),float(image[0x0020, 0x0032].value[2])])


# Convert the list of files and position coords to a dataframe
df = pd.DataFrame(data=files, columns=('image','plane','iop_x','iop_y','iop_z'))
df.head(10)