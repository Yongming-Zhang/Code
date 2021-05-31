import nibabel as nib
import numpy as np 
import os
from mitok.utils.mdicom import SERIES


file_prefix = 'mask_vessel'
#mask_nii = nib.load('/data2/zhangyongming/cpr/data_self/masks/0784361_4496F654_CTA/mask_source/'+file_prefix+".nii.gz")
mask_nii = nib.load('/data1/inputdata/1004812/058AD1E9/259A3499_CTA/mask_source/'+file_prefix+".nii.gz")
#masknorm_nii = nib.load('/data1/inputdata/1004812/058AD1E9/259A3499_CTA/masknorm_vessel.nii.gz')
mask_root = '/data2/zhangyongming/cpr/'
dcm_folder = '/data1/inputdata/1004812/058AD1E9/259A3499'
series = SERIES(series_path=dcm_folder, strict_check_series=True)
nii = np.eye(4)
print(series.series_uid)
nii[0][0] = series.pixel_spacing[0] 
nii[1][1] = series.pixel_spacing[1] 
nii[2][2] = series.slice_spacing
print(nii)
mask = mask_nii.get_fdata()
mask = np.flip(mask,axis=2)
print(mask)
print(mask.shape)
mask = np.where(mask>0,1000,mask)
print(mask.shape)
vessel_nii = nib.Nifti1Image(mask, nii)
nib.save(vessel_nii, os.path.join(mask_root, 'mask_vessel.nii.gz'))


