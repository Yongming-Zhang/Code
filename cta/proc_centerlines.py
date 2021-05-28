from cta_util import *
import glob
from lib.utils.mio import dicom_load, json_load, json_save, mkdir_safe


def gen_centerline_dict(ps_dcm_dirs, save_dir, cpr_str, using_series_id=False):
    mkdir_safe(save_dir)
    for ps_dcm_dir in ps_dcm_dirs:
        psid_items = ps_dcm_dir.split('/')[-2].split('_')
        if len(psid_items) == 3:
            pid, study, series = psid_items
        else:
            pid, study = psid_items
            series = None
        c2d_paths = sorted(glob.glob(ps_dcm_dir + cpr_str + '/*_cpr.2d'))
        centerline_dict = {
            'instance_vessel_dict': {},
            'instance_degree_dict': {},
            'instance_centerline2d_dict': {},
            'centerline3d_dict': {}
        }
        for c2d_path in c2d_paths:
            c2d_dict_list = json_load(c2d_path)
            vessel_name = c2d_path.split('/')[-1].split('_')[0]
            c3d_dict = json_load(ps_dcm_dir + cpr_str + '/%s.3d' % vessel_name)
            centerline_dict['instance_centerline2d_dict'][vessel_name] = {}
            centerline_dict['centerline3d_dict'][vessel_name] = c3d_dict['points']
            for c2d_dict in c2d_dict_list['vessel_group']:
                vessel_degree_name = c2d_dict['name']
                degree = int(vessel_degree_name.split('_')[1])
                c2d_pts = c2d_dict['points']
                dcm = dicom_load(ps_dcm_dir + '%s/%s_%03d.dcm' % (vessel_name, vessel_name, degree))
                pid = dcm.PatientID
                pid = pid.split('_')[0]
                study = dcm.StudyInstanceUID
                instance_num = dcm.InstanceNumber
                series_instance = str(dcm.SeriesInstanceUID) + '_' + str(dcm.InstanceNumber)
                centerline_dict['instance_centerline2d_dict'][vessel_name][instance_num] = c2d_pts
                centerline_dict['instance_vessel_dict'][series_instance] = vessel_name
                centerline_dict['instance_degree_dict'][series_instance] = degree
        if pid is None or study is None:
            continue
        ps_id = pid + '_' + study + '_' + series if using_series_id else pid + '_' + study
        print('processing %s, %d files, %d vessels' % (
            ps_id, len(centerline_dict['instance_vessel_dict']), len(centerline_dict['centerline3d_dict'])))
        json_save(save_dir + ps_id + '.json', centerline_dict)


def gen_centerline_dict_old_b1():
    dcm_dir = '/breast_data/cta/dicom/'
    save_dir = '/breast_data/cta/centerline_dicts/'
    ps_dcm_dirs = sorted(glob.glob(dcm_dir + '*/*/'))
    gen_centerline_dict(ps_dcm_dirs, save_dir, 'cpr/centerline')


def gen_centerline_dict_new():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    cpr_name = 'cpr_s9_n20_reorg_dcm_bmp'
    save_dir = base_dir + 'centerline_dicts/'
    ps_dcm_dirs = sorted(glob.glob('%s%s/*/' % (base_dir, cpr_name)))
    gen_centerline_dict(ps_dcm_dirs, save_dir, 'centerline')


def gen_centerline_dict_b3():
    base_dir = '/breast_data/cta/new_data/b3_azcto_150/'
    cpr_name = 'cpr_s9_n20_reorg'
    save_dir = base_dir + 'centerline_dicts/'
    ps_dcm_dirs = sorted(glob.glob('%s%s/*/' % (base_dir, cpr_name)))
    gen_centerline_dict(ps_dcm_dirs, save_dir, 'centerline')


if __name__ == '__main__':
    # gen_centerline_dict_old_b1()
    # gen_centerline_dict_new()
    gen_centerline_dict_b3()

