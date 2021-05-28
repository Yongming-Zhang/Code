from cta_util import *
import glob
import shutil
import pydicom
import nibabel as nib
from seg.skeleton import split_skeleton_segments
from lib.utils.mio import load_string_list, save_string_list, mkdir_safe, json_load, json_save, dicom_load, pickle_save


def parse_excel_as_array(xls):
    sheet = xls.sheet_by_name(xls.sheet_names()[0])
    array_list = []
    for i in range(sheet.nrows):
        row = sheet.row_values(i)
        line = [str(row[j]) for j in range(0, len(row))]
        array_list.append(line)
    return array_list


def load_pid_ref_xls(paths):
    import xlrd
    old_new_dict = {}
    for path in paths:
        lines = parse_excel_as_array(xlrd.open_workbook(path))
        old_new_dict.update({old_id: new_id for new_id, old_id in lines[1:]})
    return old_new_dict


def load_pid_ref(path):
    lines = load_string_list(path)
    old_new_dict = {}
    for line in lines:
        old_id, new_id = line.split(' ')
        old_new_dict[old_id] = new_id
    return old_new_dict


def proc_b1():
    base_dir = '/data2/zhangfd/data/cta/std_test/b1_150/'
    raw_ann_dir = base_dir + 'ann/raw_rvw_0602/gz/'
    proc_ann_dir = base_dir + 'ann/v0602/'
    mkdir_safe(proc_ann_dir)
    dst_dcm_dir = base_dir + 'dicom/'
    mkdir_safe(dst_dcm_dir)
    # ann_nii_paths = glob.glob(raw_ann_dir + '150_*.gz') + glob.glob(raw_ann_dir + '63_*.gz')
    ann_nii_paths = glob.glob(raw_ann_dir + '*.gz')
    # old_new_dict = load_pid_ref_xls([base_dir + 'config/pid_ref.az.xls', base_dir + 'config/pid_ref.wj.xls'])
    old_new_dict = load_pid_ref(base_dir + 'config/old_new_map.txt')
    pid_s8_dict = {}
    for series_dir in sorted(glob.glob(base_dir + 'raw_dicom/*/*/*/')):
        pid, study, series = series_dir.split('/')[-4:-1]
        pid_s8_dict[pid + '_' + series[-8:]] = [pid, study, series]
    sub_dirs, sub_dir_old_names = [], []
    for ann_nii_path in ann_nii_paths:
        ann_name = ann_nii_path.split('/')[-1]
        if 'PID' in ann_name:
            ann_name = ann_name.replace('-', '_')
        old_pid, series8 = ann_name.split('_')[1:3]
        if old_pid not in old_new_dict:
            print('MISSING ID:', old_pid)
            continue
        new_pid = old_new_dict[old_pid]
        pid_s8 = new_pid + '_' + series8
        if pid_s8 not in pid_s8_dict:
            wj_dir = '/data2/ltq/instance/test_coronary/wujing/wj/'
            dcm_paths = glob.glob(wj_dir + '%s/*/%s/*.dcm' % (old_pid, series8))
            if len(dcm_paths) > 0:
                dcm = dicom_load(dcm_paths[0])
                series8 = dcm.SeriesInstanceUID[-8:]
                pid_s8 = new_pid + '_' + series8
        if pid_s8 in pid_s8_dict:
            pid, study, series = pid_s8_dict[pid_s8]
            sub_dir = pid + '_' + study + '_' + series
            sub_dirs.append(sub_dir)
            sub_dir_old_names.append(sub_dir + ' ' + ann_nii_path.split('/')[-1])
            shutil.copy(ann_nii_path, proc_ann_dir + sub_dir + '.nii.gz')
            shutil.copytree(base_dir + 'raw_dicom/%s/%s/%s/' % (pid, study, series), dst_dcm_dir + sub_dir + '/')
            print('%s %s' % (sub_dir, ann_nii_path.split('/')[-1]))
        else:
            print(old_pid, pid_s8)
    print(len(sub_dirs))
    save_string_list(base_dir + 'config/sub_dirs.txt', sorted(sub_dirs))
    save_string_list(base_dir + 'config/sub_dirs_with_old.txt', sorted(sub_dir_old_names))


def get_slice_spacing(dcm1, dcm2):
    dz = float(dcm1.SliceLocation) - float(dcm2.SliceLocation)
    di = int(dcm1.InstanceNumber) - int(dcm2.InstanceNumber)
    return dz / di


def gen_data_info(dcm_dir, sub_dirs):
    data_info_dict = {}
    for sub_dir in sub_dirs:
        sub_dcm_dir = dcm_dir + sub_dir + '/'
        dcm_paths = sorted(glob.glob(sub_dcm_dir + '*.dcm'))
        dcm = dicom_load(dcm_paths[0])
        pid, study, series = sub_dir.split('_')
        man = dcm.Manufacturer.split(' ')[0]
        kernel = dcm.ConvolutionKernel
        if isinstance(kernel, pydicom.multival.MultiValue):
            kernel = kernel[0]
        slice_spacing = get_slice_spacing(dcm, dicom_load(dcm_paths[1]))
        spacing_x, spacing_y = dcm.PixelSpacing
        spacings = (float(spacing_x), float(spacing_y), abs(slice_spacing))
        age = int(dcm.PatientAge.lower().replace('y', ''))
        data_info_dict[sub_dir] = {
            'man': man,
            'kernel': kernel,
            'spacings': spacings,
            'age': age,
            'study': study,
            'series': series,
            'dcm_num': len(dcm_paths),
            'flip': slice_spacing > 0
        }
        print('%s: %s, %s, %s, %d' % (pid, man, kernel, spacings, age))
    print(len(data_info_dict))
    return data_info_dict


def gen_data_info_main():
    base_dir = '/data2/zhangfd/data/cta/std_test/b1_150/'
    dcm_dir = base_dir + 'dicom/'
    sub_dirs = load_string_list(base_dir + 'config/sub_dirs.txt')
    json_save(base_dir + 'config/data_info.json', gen_data_info(dcm_dir, sub_dirs))


def proc_b2():
    base_dir = '/data2/zhangfd/data/cta/std_test/b2_119/'
    src_dcm_dir = base_dir + 'raw_dicom/'
    sub_dirs = []
    for sub_dcm_dir in sorted(glob.glob(src_dcm_dir + '*/*/*/')):
        dcm_paths = glob.glob(sub_dcm_dir + '*.dcm')
        if len(dcm_paths) > 100:
            pid, study, series = sub_dcm_dir.split('/')[-4:-1]
            sub_dirs.append(pid + '_' + study + '_' + series)
    save_string_list(base_dir + 'config/sub_dirs.txt', sorted(sub_dirs))
    print(len(sub_dirs))


def post_process_vessel_mask():
    base_dir = '/data2/zhangfd/data/cta/std_test/b1_150/'
    sub_dirs = load_string_list(base_dir + 'config/sub_dirs.txt')
    ann_dir = base_dir + 'ann/v0602/'
    save_dir = ann_dir + 'post/'
    mkdir_safe(save_dir)
    for sub_dir in sub_dirs:
        vm_path = ann_dir + sub_dir + '.nii.gz'
        gt_nii = nib.load(vm_path)
        gt = gt_nii.get_data()
        ske_pts, seg_pts, seg_types, labels = split_skeleton_segments(gt)
        post_res = {
            'ske_pts': ske_pts,
            'seg_pts': seg_pts,
            'seg_types': seg_types
        }
        pickle_save(save_dir + sub_dir + '.pkl', post_res)
        nib.save(nib.Nifti1Image(labels, gt_nii.affine), save_dir + sub_dir + '.nii.gz')
        print('%s' % sub_dir)
        seg_list = [(len(seg), seg_type) for seg, seg_type in zip(seg_pts, seg_types) if len(seg) >= 20]
        print('\t%d pts, %d/%d segs: %s' % (ske_pts.shape[0], len(seg_pts), len(seg_list), sorted(seg_list)))


if __name__ == '__main__':
    # proc_b1()
    # gen_data_info_main()
    # post_process_vessel_mask()
    proc_b2()

