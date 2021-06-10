import os
import glob
import shutil

file_dir = '/data2/zhangyongming/cpr/ffr_dataset'
files = os.listdir(file_dir)
for name in files:
    study_name = glob.glob(os.path.join(file_dir, name, '*'))[0]
    study_list = os.listdir(study_name)
    for study_file in study_list:
        if study_file.split('_')[-1] == 'CTA':
            print(study_name+'/'+study_file)
            shutil.rmtree(os.path.join(study_name, study_file))
