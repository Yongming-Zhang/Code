#-*-coding:utf-8-*-
import pydicom
import json
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import skimage.io as io
from mitok.utils.mdicom import SERIES
import nibabel as nib
import numpy as np
import os
import glob
import shutil

#校对标注前，要对dicom进行判断，来确定nii.gz是否要翻转
dataset_list = [1022832, 1022839, 1036601, 1036604, 1036612, 1073305, 1073313, 1073317, 1073318, 1073332]
dicom_path = '/data1/zhangyongming/dataset/dicom/'
nii_path = '/data1/zhangyongming/ffr_plaque_mask/'
dicom_path_list = os.listdir(dicom_path)
for patient in dicom_path_list:
    patient_dir = glob.glob(os.path.join(dicom_path, patient, '*'))[0]
    series_list = os.listdir(patient_dir)
    for series_id in series_list:        
        dcm_folder = os.path.join(patient_dir, series_id)
        print(dcm_folder)
        #if dcm_folder.split('_')[-1] == 'CTA':
            #shutil.rmtree(dcm_folder)
        #    continue
        series = SERIES(series_path=dcm_folder, strict_check_series=True)
        print(series.patient_id, series.flip)

        #判断series.flip是否为false，若为false，则对z轴做flip，若为true，则不需要flip
        if not series.flip:
            data_dir = nii_path + patient
            plaque_path = os.path.join(data_dir, series_id, 'mask_plaque.nii.gz')
            img = sitk.ReadImage(plaque_path)
            data = sitk.GetArrayFromImage(img)
            print(data.shape)
            affine_arr = np.eye(4)
            data = data[::-1,:,:]
            data = np.transpose(data, (2, 1, 0))
            #data = data.astype('float32')
            plaque_nii = nib.Nifti1Image(data, affine_arr)
            new_path = os.path.join('/data1/zhangyongming/transformed_z_plaque_new', patient, series_id)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            nib.save(plaque_nii, os.path.join(new_path, 'mask_plaque.nii.gz')) 
        
        else:
            data_dir = nii_path + patient
            plaque_path = os.path.join(data_dir, series_id, 'mask_plaque.nii.gz')
            new_path = os.path.join('/data1/zhangyongming/transformed_z_plaque_new', patient, series_id)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            shutil.copy(plaque_path, os.path.join(new_path))
