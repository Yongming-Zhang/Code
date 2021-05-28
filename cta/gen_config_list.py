from cta_util import *
import glob
from lib.utils.mio import *


def gen_b2_lists():
    # src_dir = '/breast_data/wangsiwen/cta/config_407/'
    base_dir = '/breast_data/cta/new_data/b3_azcto_150/'
    dst_dir = base_dir + 'config/'
    psid_list = sorted([d.split('/')[-2] for d in glob.glob(base_dir + 'cta_dicom/*/')])
    save_string_list(dst_dir + 'psid_list.txt', psid_list)
    # pid_list = sorted(list(set([psid.split('_')[0] for psid in psid_list])))
    # save_string_list(dst_dir + 'pid_list.txt', pid_list)
    for set_name in ['train', 'test', 'valid']:
        # set_pid_list = set(load_string_list(src_dir + set_name + '_b2_407.txt'))
        set_pid_list = load_string_list(dst_dir + set_name + '_pid_list.txt')
        full_set_name = 'valid' if set_name == 'val' else set_name
        set_psid_list = []
        for psid in psid_list:
            if psid.split('_')[0] in set_pid_list:
                set_psid_list.append(psid)
        save_string_list(dst_dir + '%s_psid_list.txt' % full_set_name, sorted(set_psid_list))
        # save_string_list(dst_dir + '%s_pid_list.txt' % full_set_name, sorted(list(set_pid_list)))


if __name__ == '__main__':
    gen_b2_lists()

