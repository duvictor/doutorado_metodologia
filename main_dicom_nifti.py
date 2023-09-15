'''
responsável por converter dicom to nifti
09/07/23
by
https://www.kaggle.com/code/rinichristy/dicom-to-nifti-conversion-using-nibabel
'''


from pathlib import Path # pathlib for easy path handling
import pydicom # pydicom to handle dicom files
import matplotlib.pyplot as plt
import numpy as np
import dicom2nifti # to convert DICOM files to the NIftI format
import nibabel as nib # nibabel to handle nifti files


dir = r"D:\dataset_cq500\CQ500CT10 CQ500CT10\Unknown Study\CT PLAIN THIN"


# head_dicom = Path(dir)
dicom2nifti.convert_directory(dir, r"D:\dataset_cq500\CQ500CT10 CQ500CT10\Unknown Study\teste.nii")


a = 45

