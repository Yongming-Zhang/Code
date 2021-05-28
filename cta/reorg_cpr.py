from cta_util import *
import glob
import shutil
import pydicom
from lib.utils.mio import *


def reorg_cpr():
    base_dir = '/breast_data/cta/new_data/b3_azcto_150/'
    src_cpr_dir = base_dir + 'cpr_lumen_s9_n20/'
    dst_cpr_dir = base_dir + 'cpr_s9_n20_reorg/'
    vessel_name_dict = load_vessel_name_map()
    for study_dir in sorted(glob.glob(src_cpr_dir + '*/cpr/')):
        dcm_paths = sorted(glob.glob(study_dir + '*.dcm'))
        pid_study = study_dir.split('/')[-3]
        for dcm_path in dcm_paths:
            dcm = pydicom.dcmread(dcm_path)
            dcm_name = dcm_path.split('/')[-1]
            vessel_name = dcm_name.split('_')[0]
            degree = int(dcm_name.split('.')[0].split('_')[1])
            if vessel_name in vessel_name_dict:
                old_ves_name = vessel_name
                vessel_name = vessel_name_dict[vessel_name]
            new_dcm_name = vessel_name + '_%03d.dcm' % degree
            dcm[(0x0020, 0x000e)].value = vessel_name
            dcm[(0x0008, 0x103e)].value = vessel_name
            save_dir = dst_cpr_dir + pid_study + '/' + vessel_name + '/'
            mkdir_safe(save_dir)
            dcm.save_as(save_dir + new_dcm_name)
            # shutil.copy(dcm_path + '.dat', save_dir + new_dcm_name + '.dat')
            shutil.copy(dcm_path + '.bmp', save_dir + new_dcm_name + '.bmp')
        vessel_num = len(glob.glob(dst_cpr_dir + pid_study + '/*/'))
        mkdir_safe(dst_cpr_dir + pid_study + '/centerline/')
        for centerline_path in sorted(glob.glob(study_dir + 'centerline/*.*d')):
            file_name = centerline_path.split('/')[-1]
            vessel_name = file_name[:-3].split('_')[0]
            if vessel_name in vessel_name_dict:
                old_ves_name = vessel_name
                vessel_name = vessel_name_dict[vessel_name]
                file_name = file_name.replace(old_ves_name, vessel_name)
            shutil.copy(centerline_path, dst_cpr_dir + pid_study + '/centerline/' + file_name)
        vn2d = len(glob.glob(dst_cpr_dir + pid_study + '/centerline/*.2d'))
        vn3d = len(glob.glob(dst_cpr_dir + pid_study + '/centerline/*.3d'))
        print('completed %s, %d/%d dcm, dir=%d/2d=%d/3d=%d vessels' % (
            pid_study, len(dcm_paths), len(glob.glob(dst_cpr_dir + pid_study + '/*/*.dcm')), vessel_num, vn2d, vn3d))


if __name__ == '__main__':
    reorg_cpr()

