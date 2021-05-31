import pydicom
import json
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import skimage.io as io
from mitok.utils.mdicom import SERIES
import os
import glob
import shutil

def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    print(dir(ds))
    print(ds.pixel_array)
    img = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(img)
    #io.imshow(img[0], cmap='gray')
    #io.show()
    #plt.imshow(ds.pixel_array)
    return information

#a = loadFileInformation('/data2/xiongxiaoliang/ffr_dataset/1036604/A4CC76BE/6566EAC0/100.dcm')
#print(a)

series = SERIES(series_path='/data2/zhangyongming/cpr/data_self/images/dcms/0785095_D2E3F722_CTA', strict_check_series=True)
print(series.patient_id, series.flip)

path = '/data1/zhangyongming/dataset/dicom/'
path_list = os.listdir(path)
for patient in path_list:
    patient_dir = glob.glob(os.path.join(path, patient, '*'))[0]
    series_list = os.listdir(patient_dir)
    for series_id in series_list:        
        dcm_folder = os.path.join(patient_dir, series_id)
        print(dcm_folder)
        if dcm_folder.split('_')[-1] == 'CTA':
            #shutil.rmtree(dcm_folder)
            continue
        series = SERIES(series_path=dcm_folder, strict_check_series=True)
        print(series.patient_id, series.flip)