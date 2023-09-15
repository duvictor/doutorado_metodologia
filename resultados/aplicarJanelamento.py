'''
19/04/2023
pegar um exame qualquer, aplicar janelamento para comparar com o resultado da rede
aplicar janelamento para tecidos moles, tecidos duros, e qualquer outros que forem uteis para a metodologia


https://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/
https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
https://www.kaggle.com/code/redwankarimsony/ct-scans-dicom-files-windowing-explained
https://radiopaedia.org/articles/windowing-ct
https://simpleitk.org/SPIE2019_COURSE/04_basic_registration.html

'''

import SimpleITK as sitk
from SimpleITK import CurvatureFlow, GetArrayFromImage, ConnectedThreshold, Cast, RescaleIntensity, LabelOverlay
from matplotlib import pyplot as plt, cm
import numpy as np




file_name = r"E:\PycharmProjects\pythonProject\exame\CQ500CT0\Unknown Study\CT PLAIN THIN\CT000135.dcm"
data_directory = r"E:\PycharmProjects\pythonProject\exame\CQ500CT0\Unknown Study\CT PLAIN THIN"

slice_exam = sitk.ReadImage(file_name)


def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    plt.show()


labelWhiteMatter = 1
labelGrayMatter = 2



# sitk_show(slice_exam[:,:,0], title='Slice#0', dpi=80)

window_level = 30
window_width = 80



windowing = sitk.IntensityWindowingImageFilter()
windowing.SetWindowMinimum(window_level)
windowing.SetWindowMaximum(window_width)
img_win = windowing.Execute(slice_exam)



imgOriginal = img_win[:,:,0]

# image = sitk.GetArrayFromImage(imgOriginal)
# image = image.astype(np.int16)
sitk_show(imgOriginal)



# Smoothing/Denoising
imgSmooth = CurvatureFlow(image1=imgOriginal, timeStep=0.125, numberOfIterations=5)

# blurFilter = SimpleITK.CurvatureFlowImageFilter()
# blurFilter.SetNumberOfIterations(5)
# blurFilter.SetTimeStep(0.125)
# imgSmooth = blurFilter.Execute(imgOriginal)

sitk_show(imgSmooth)

lstSeeds = [(150,75)]

imgWhiteMatter = ConnectedThreshold(image1=imgSmooth,
                                              seedList=lstSeeds,
                                              lower=130,
                                              upper=190,
                                              replaceValue=labelWhiteMatter)


# Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
imgSmoothInt = Cast(RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())

# Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatter'
sitk_show(LabelOverlay(imgSmoothInt, imgWhiteMatter))

a = 45
b = 90