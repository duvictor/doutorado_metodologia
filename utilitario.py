import pydicom
import cv2






# Convert DICOM to JPG/PNG via openCV
def convert_images(filename, outdir, img_type='jpg'):
    """Reads a dcm file and saves the files as png/jpg

    Args:
        filename: path to the dcm file
        img_type: format of the processed file (jpg or png)
    https://discuss.pytorch.org/t/dicom-files-in-pytorch/49058/5
    """
    # extract the name of the file
    name = filename.parts[-1]

    # read the dcm file
    ds = pydicom.read_file(str(filename))
    img = ds.pixel_array

    # save the image as jpg/png
    if img_type=="jpg":
        cv2.imwrite(outdir + name.replace('.dcm' ,'.jpg'), img)
    else:
        cv2.imwrite(outdir + name.replace('.dcm' ,'.png'), img)

# Using dask
# all_images = [dd.delayed(convert_images)(all_files[x]) for x in range(len(all_files))]
# dd.compute(all_images)