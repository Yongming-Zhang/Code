import json
import numpy
import glob
import os

def map_cta_ann_json_to_cpr():
    cta_json_src_dict = '/mnt/users/code/cta/task_1985_0-1613536411.json'
    cta_json_tar_dict = '/mnt/users/code/cta/task_1985_0-1613536411_new.json'
    cta_path = '/data1/zhangfd/data/cta/b4_lz493/cta_dicom/'
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
                if isinstance(node['descText'][1][0][0]['select'], list):
                    continue
                les_type = node['descText'][1][0][0]['select']
                stenosis = node['descText'][1][1][0]['select']
                detect_necessity = node['descText'][1][2][0]['select']
                segments = ['other' if s == u'其他' else s for s in node['descText'][1][3][0]['select']]
                if les_type == 'none':
                    continue
                new_node = {'type': les_type, 'detect_necessity': detect_necessity, 'segments': segments}
                if les_type in ['stent_clear', 'stent_unclear', 'stent_block', 'stent_none', 'mca', 'mca_mb']:
                    pts3d = []
                    for roi in node['rois']:
                        z = node[roi['slice_index']]
                        x, y = roi['edge'][0]
                        pts3d.append([x, y, z])
                    new_node['pts3d'] = pts3d
                else:
                    new_node['stenosis'] = stenosis
                    cta_list = os.listdir(os.path.join(cta_path))
                    for series in cta_list:
                        if data['patientID'] == series.split('_')[0]:
                            cta_paths = glob.glob(os.path.join(cta_path, series)
                    cta_paths = glob.glob(os.path.join(cta_paths, '*.dcm'))
                    cta_paths.sort(key=lambda ele: int(ele.split('/')[-1].split('.')[0]))
                    cta_mask = numpy.zeros(cta_shape, dtype='uint8')
                    for roi in node['rois'] + node['bounds']:
                        contours_numpy = numpy.int32(roi['edge'])
                        z = node[roi['slice_index']]
                        cv2.drawContours(cta_mask[z], (contours_numpy,), 0, color=255, thickness=-1)
                    cpr_masks = mapper.cta2cpr(cta_mask, 'trilinear')
                    rois = []
                    for i, cpr_mask in enumerate(cpr_masks):
                        edge_list = find_contours(cpr_mask)
                        for edge in edge_list:
                            rois.append({'slice_index': node[i], 'edge': edge[:, 0, :].tolist()})
                    if len(rois) == 0:
                        new_node = None
                    else:
                        new_node['rois'] = rois_node['pts3d'] = pts3d
                
                if new_node is not None:
                    cpr_json_dict['nodes'].append(new_node)
        
            json.dump(cpr_json_dict, w)
            w.write('\n')

if __name__ == '__main__':
    map_cta_ann_json_to_cpr()
