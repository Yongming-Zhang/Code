#-*- coding:utf-8 -*-
import SimpleITK as sitk 
import pydicom
from pydicom import dcmread, dicomio 

# path_read:读取dicom的文件路径  path_save:保存nii的文件路径
def dcm2nii(path_read, path_save):
    # GetGDCMSeriesIDs读取序列号相同的dcm文件
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    ds = dicomio.read_file('/data2/zhangyongming/cpr/data_self/images/dcms/1004812_259A3499_CTA/N2D_0099.dcm')
    print('tag1',ds.dir())
    dr = dicomio.read_file('/data1/inputdata/1004812/058AD1E9/259A3499/99.dcm')
    print('tag2',dr.dir())
    # GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    print(len(series_file_names))
    #series_reader = sitk.ImageSeriesReader()
    #series_reader.SetFileNames(series_file_names)
    #image3d = series_reader.Execute()
    #sitk.WriteImage(image3d, path_save)

def read_dicom(path_dicm):
    filename = path_dicm + '/N2D_0100.dcm'
    data = dcmread(filename)
    print(data.pixel_array)
    print(data)

if __name__ == '__main__':
    path_dicm = '/data2/zhangyongming/cpr/outputs_self/1004812_259A3499_CTA/scpr'
    path_nii = './image/test.nii.gz'
    path_dicm1 = '/data2/zhangyongming/cpr/image'
    dcm2nii(path_dicm, path_nii)
    #read_dicom(path_dicm1)
