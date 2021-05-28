from cta_util import *
from lib.utils.mio import json_load
import glob
import numpy


def load_c3d_as_numpy(c3d_path):
    return numpy.float32(json_load(c3d_path)['points'])


def check_old_new_centerline_3d():
    old_dir = '/breast_data/cta/dicom/'
    new_dir = '/breast_data/cta/new_data/b1_sg_587/cpr_s18_n10_reorg_dcm/'
    new_ps_dirs = glob.glob(new_dir + '*/')
    pid_ps_dict = {}
    for d in new_ps_dirs:
        ps = d.split('/')[-2]
        pid, study = ps.split('_')
        assert pid not in pid_ps_dict
        pid_ps_dict[pid] = ps
    old_c3d_paths = sorted(glob.glob(old_dir + '*/*/cpr/centerline/*.3d'))
    delta_list = []
    for old_path in old_c3d_paths:
        items = old_path.split('/')
        pid, c3d_name = items[-4].split('_')[0], items[-1]
        new_path = new_dir + pid_ps_dict[pid] + '/centerline/' + c3d_name
        c3d_old = load_c3d_as_numpy(old_path)
        c3d_new = load_c3d_as_numpy(new_path)
        c3d_delta = numpy.abs(c3d_old - c3d_new)
        delta_mean_x, delta_mean_y, delta_mean_z = c3d_delta.mean(axis=0)
        delta_max_x, delta_max_y, delta_max_z = c3d_delta.max(axis=0)
        print('%s, d_mean=(%.2f, %.2f, %.2f), d_max=(%.2f, %.2f, %.2f)' % (
            new_path, delta_mean_x, delta_mean_y, delta_mean_z, delta_max_x, delta_max_y, delta_max_z))
        delta_list.append([delta_mean_x, delta_mean_y, delta_max_x, delta_max_y])
    delta_list = numpy.float32(delta_list)
    print(delta_list.max(axis=0))
    print(delta_list.mean(axis=0))


if __name__ == '__main__':
    check_old_new_centerline_3d()

