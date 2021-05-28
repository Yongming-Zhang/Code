from cta_util import *
import glob
import pydicom
from collections import defaultdict
from global_var import plaque_type_all_dict
from lib.utils.mio import json_load, load_string_list, mkdir_safe, json_save


def load_b1_raw_ann():
    task_names = ['task_10626_0', 'task_10652_0-1582016734', 'task_10627_0-1586145902']
    jd_list = []
    for task_name in task_names:
        jd_list += json_load('/breast_data/cta/annotation/%s.json' % task_name)
    return jd_list


def pts3d_dicts_2_ann_json(pts3d_dicts, cta_dir, save_dir):
    psids = sorted(pts3d_dicts.keys())
    for psid in psids:
        pts3d_list = pts3d_dicts[psid]
        cta_paths = glob.glob(cta_dir + '/' + psid + '/*.dcm')
        instance_nums = sorted([int(cta_path.split('/')[-1][:-4]) for cta_path in cta_paths])
        dcm = pydicom.dcmread(cta_paths[0])
        pid, study, series = dcm.PatientID, dcm.StudyInstanceUID, dcm.SeriesInstanceUID
        jd = {
            'patientID': pid,
            'studyUID': study,
            'seriesUID': series,
            'other_info': {
                'instance_range': (instance_nums[0], instance_nums[-1])
            },
            'nodes': []
        }
        pss_save_dir = save_dir + '/' + pid + '/' + study + '/' + series + '/'
        mkdir_safe(pss_save_dir)
        for node_index, p in enumerate(pts3d_list):
            new_node = {
                'node_index': node_index,
                'rois': [],
                'merged_type': p.les_type
            }
            for x, y, z in p.pts3d:
                new_node['rois'].append({'slice_index': instance_nums[z], 'edge': [[int(x), int(y)]]})
            jd['nodes'].append(new_node)
        json_save(pss_save_dir + 'ann.json', jd)
        print('%s: %d nodes' % (pss_save_dir, len(jd['nodes'])))


def b1_main():
    base_dir = '/breast_data/cta/new_data/b1_bt_587/'
    centerline_dir = '/breast_data/cta/centerline_dicts/'
    raw_jd_list = load_b1_raw_ann()
    ps_list = load_string_list(base_dir + 'config/psid_list.txt')
    # ps_list = ['0657753_1.3.12.2.1107.5.1.4.75509.20180528074202710786']
    ps_cl_dicts = {ps: json_load(centerline_dir + ps + '.json') for ps in ps_list}
    ps_plaque_list = {}
    les_num = 0
    for jd in raw_jd_list:
        ps = jd['patientID'].split('_')[0] + '_' + jd['studyUID']
        if ps not in ps_cl_dicts:
            continue
        if ps not in ps_plaque_list:
            ps_plaque_list[ps] = []
        cl_dict = ps_cl_dicts[ps]
        for node in jd['nodes']:
            les_type = node['descText'][1][0][0]['select']
            if les_type not in plaque_type_all_dict:
                continue
            les_type = plaque_type_all_dict[les_type]
            for roi in node['rois'] + node['bounds']:
                size_w, size_h = get_size(roi['edge'])
                instance_num = str(roi['slice_index'])
                c2d_pts = numpy.array(cl_dict['instance_centerline2d_dict'][instance_num])
                vessel_name = cl_dict['instance_vessel_dict'][instance_num]
                c3d_pts = numpy.array(cl_dict['centerline3d_dict'][vessel_name])
                pts3d = map_roi_to_3d(roi, c2d_pts, c3d_pts, stride=1)
                if pts3d is None:
                    # print(ps, les_type, vessel_name, instance_num)
                    # print(c2d_pts)
                    # print(roi)
                    continue
                ps_plaque_list[ps].append(Plaque3D(pts3d, les_type, (size_w + size_h) * 0.5))
    print('merging nodes ...')
    merged_plaque3d_list_dict = {ps: Plaque3D.merge(plaque_list) for ps, plaque_list in ps_plaque_list.items()}
    print('generating json dicts ...')
    pts3d_dicts = {}
    les_type_les_nums = defaultdict(lambda: 0)
    for ps, merged_plaque_list in merged_plaque3d_list_dict.items():
        pts3d_dicts[ps] = []
        for p in merged_plaque_list:
            les_type_les_nums[p.les_type] += 1
            les_num += 1
            pts3d_dicts[ps].append(p)
    pts3d_dicts_2_ann_json(pts3d_dicts, base_dir + 'cta_dicom/', base_dir + 'annotations/ann_pts3d_for_check/')
    case_num = len(ps_plaque_list)
    les_num_str = ', '.join(['%s=%d' % (les_type, num) for les_type, num in les_type_les_nums.items()])
    print('processed %d case, including %d (%s) lesions' % (case_num, les_num, les_num_str))


def is_ann_by_endpoint(les_type):
    return les_type in ['stent_clear', 'stent_unclear', 'stent_block', 'stent_none', 'mca', 'mca_mb']


def b2_main():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    save_dir = base_dir + 'annotation/ann3d_for_check/'
    raw_jd_list = json_load(base_dir + 'annotation/task_1265_0-1588272710.json')
    for jd in raw_jd_list:
        sub_save_dir = save_dir + jd['patientID'] + '/' + jd['studyUID'] + '/' + jd['seriesUID'] + '/'
        mkdir_safe(sub_save_dir)
        new_nodes = []
        for node_index, node in enumerate(jd['nodes']):
            if isinstance(node['descText'][1][0][0]['select'], list):
                continue
            les_type = g_ccta_lesion_type_dict[node['descText'][1][0][0]['select']]
            new_node = {
                'node_index': node_index,
                'rois': node['rois'] + node['bounds']
            }
            if is_ann_by_endpoint(les_type):
                rois = []
                for roi in node['rois']:
                    rois.append({
                        'slice_index': roi['slice_index'],
                        'edge': roi['edge'][:1],
                    })
                new_node['rois'] = rois
            new_nodes.append(new_node)
        jd['nodes'] = new_nodes
        json_save(sub_save_dir + 'ann.json', jd)


if __name__ == '__main__':
    b1_main()
    # b2_main()

