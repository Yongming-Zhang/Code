import json
import numpy

def map_cta_ann_json_to_cpr():
    cta_json_src_dict = '/mnt/users/code/cta/task_1985_0-1613536411.json'
    cta_json_tar_dict = '/mnt/users/code/cta/task_1985_0-1613536411_new.json'
    with open(cta_json_src_dict, 'r') as f:
        datas = json.load(f)
    with open(cta_json_tar_dict, 'w') as w:
        for data in datas:
            cpr_json_dict = {
                'patientID': data['patientID'],
                'studyUID': data['studyUID'],
                'other_info': {'doctorId': data['other_info']['doctorId']},
                'nodes': []
            }
            for node in data['nodes']:
                cpr_json_dict['nodes'].append({'node_index':node['node_index']})
                cpr_json_dict['nodes'].append({'rois':node['rois']})
                cpr_json_dict['nodes'].append({'bounds':node['bounds']})
                if isinstance(node['descText'][1][0][0]['select'], list):
                    continue
                les_type = node['descText'][1][0][0]['select']
                stenosis = node['descText'][1][1][0]['select']
                detect_necessity = node['descText'][1][2][0]['select']
                segments = ['other' if s == u'其他' else s for s in node['descText'][1][3][0]['select']]
                if les_type == 'none':
                    continue
                new_node = {'type': les_type, 'detect_necessity': detect_necessity, 'segments': segments}
                pts3d = []
                for roi in node['rois']:
                    z = roi['slice_index']
                    x, y = roi['edge'][0]
                    pts3d.append([x, y, z])
                new_node['pts3d'] = pts3d
                
                if new_node is not None:
                    cpr_json_dict['nodes'].append(new_node)
        
            json.dump(cpr_json_dict, w)
            w.write('\n')

if __name__ == '__main__':
    map_cta_ann_json_to_cpr()