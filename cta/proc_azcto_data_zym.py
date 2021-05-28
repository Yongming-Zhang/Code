from cta_util import *
from lib.utils.mio import mkdir_safe, json_save


segment_vessel_dict = {
    'pRCA': ['RCA'],
    'mRCA': ['RCA'],
    'dRCA': ['RCA'],
    'R-PDA': ['RCA', 'RPDA'],
    'LM': ['LAD'],
    'pLAD': ['LAD'],
    'mLAD': ['LAD'],
    'dLAD': ['LAD'],
    'pCx': ['LCX'],
    'LCx': ['LCX'],
}


def create_new_cpr_json(cta_ann_json_dict):
    cpr_json_dict = {
        'patientID': cta_ann_json_dict['patientID'],
        'studyUID': cta_ann_json_dict['studyUID'],
        'other_info': {'doctorId': cta_ann_json_dict['other_info']['doctorId']},
        'nodes': []
    }
    return cpr_json_dict


def map_cta_ann_to_cpr(cta_ann_json_dict, cta_instance_dict, existed_vessel_names):
    cpr_json_dicts = {}
    for node in cta_ann_json_dict['nodes']:
        if isinstance(node['descText'][1][0][0]['select'], list):
            continue
        les_type = g_ccta_lesion_type_dict[node['descText'][1][0][0]['select']]
        stenosis = g_stenosis_type_dict[node['descText'][1][1][0]['select']]
        segments = node['descText'][1][2][0]['select']
        if les_type == 'none':
            continue
        new_node = {'type': les_type, 'segments': segments, 'stenosis': stenosis}
        pts3d = []
        for roi in node['rois']:
            z = cta_instance_dict[roi['slice_index']]
            x, y = roi['edge'][0]
            pts3d.append([x, y, z])
        new_node['pts3d'] = pts3d
        vessel_names = []
        for segment in segments:
            for vessel_name in segment_vessel_dict.get(segment, segment.replace('-', '')):
                if vessel_name in existed_vessel_names:
                    vessel_names.append(vessel_name)
                    break
        for vessel_name in list(set(vessel_names)):
            if vessel_name not in cpr_json_dicts:
                cpr_json_dicts[vessel_name] = create_new_cpr_json(cta_ann_json_dict)
            cpr_json_dicts[vessel_name]['nodes'].append(new_node)
    return cpr_json_dicts


def gen_cta_pssid_json_dict(cta_json_dict_list):
    cta_pssid_json_dicts = {}
    for cta_json_dict in cta_json_dict_list:
        if cta_json_dict['other_info']['passReason'] != '':
            continue
        if len(cta_json_dict['nodes']) == 0:
            continue
        pssid = cta_json_dict['patientID'] + '_' + cta_json_dict['studyUID'] + '_' + cta_json_dict['seriesUID']
        if pssid not in cta_pssid_json_dicts:
            cta_pssid_json_dicts[pssid] = cta_json_dict
        else:
            cta_pssid_json_dicts[pssid]['nodes'] += cta_json_dict['nodes']
    return cta_pssid_json_dicts


def gen_cpr_json_main():
    #base_dir = '/breast_data/cta/new_data/b3_azcto_150/'
    cpr_dir = '/data1/zhangyongming/cta/b4_lz493/cpr_scpr_lumen_s18_n10/'#base_dir + 'cpr_s9_n20_reorg/'
    cta_dcm_dir = '/data1/zhangfd/data/cta/b4_lz493/cta_dicom/' #base_dir + 'dicom/'
    #cta_ann_json_dir = base_dir + 'annotation/'
    cpr_json_dir = '/data1/zhangyongming/new_json/'#cta_ann_json_dir + 'cpr_s9_n20_reorg/raw_json/'
    mkdir_safe(cpr_json_dir)
    cta_json_dict_list = json_load('/data1/zhangyongming/' + 'task_1830_0-1599383101.json')
    cta_pssid_json_dicts = gen_cta_pssid_json_dict(cta_json_dict_list)
    print('%d pssid' % len(cta_pssid_json_dicts))
    for pssid in sorted(cta_pssid_json_dicts.keys()):
        cta_json_dict = cta_pssid_json_dicts[pssid]
        cpr_series_dirs = sorted(glob.glob(cpr_dir + pssid + '/*/'))
        cta_dcm_paths = sorted(glob.glob(cta_dcm_dir + pssid + '/*.dcm'))
        cta_instance_dict = {int(p.split('/')[-1][:-4]): i for i, p in enumerate(cta_dcm_paths)}
        existed_vessel_names = []
        for cpr_series_dir in cpr_series_dirs:
            vessel_name = cpr_series_dir.split('/')[-2]
            if vessel_name != 'centerline':
                existed_vessel_names.append(vessel_name)
        cpr_json_dicts = map_cta_ann_to_cpr(cta_json_dict, cta_instance_dict, set(existed_vessel_names))
        for vessel_name, cpr_json_dict in cpr_json_dicts.items():
            if len(cpr_json_dict['nodes']) == 0:
                continue
            print('%s/%s: %d cta nodes, %d cpr nodes' % (
                pssid, vessel_name, len(cta_json_dict['nodes']), len(cpr_json_dict['nodes'])))
            json_save(cpr_json_dir + pssid + '_' + vessel_name + '.json', cpr_json_dict)


if __name__ == '__main__':
    gen_cpr_json_main()

