from lumen_util import *
import shutil
import glob
from lib.utils.mio import *


def copy_cpr_lumen():
    base_dir = '/data2/zhangfd/data/cta/b2_407/cta2cpr/'
    src_dir = base_dir + 'cpr_cta_lumen_s9_n20_reorg/'
    dst_dir = base_dir + 'cpr_lumen_s9_n20_reorg/'
    src_cpr_lumen_dirs = sorted(glob.glob(src_dir + '*/CPR_*/') + glob.glob(src_dir + '*/LUMEN/'))
    for sd in src_cpr_lumen_dirs:
        psid, cpr_lumen_str = sd.split('/')[-3:-1]
        dd = dst_dir + psid + '/'
        mkdir_safe(dd)
        shutil.copytree(sd, dd + cpr_lumen_str + '/')


if __name__ == '__main__':
    copy_cpr_lumen()

