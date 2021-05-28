from cta_cpr_map import *
from lib.utils.mio import *
import glob
import numpy
from lib.image.image import gray2rgb
from lib.image.draw import draw_texts
import pydicom
from collections import defaultdict

#def gen_cta_psid_json_dict(cta_json_dict_list):
#    cta_psid_json_dicts = {}
#    for cta_json_dict in cta_json_dict_list:
#        if cta_json_dict['other_info']['passReason'] != '':
#            continue
#        if len(cta_json_dict['nodes']) == 0:
#            continue
#        psid = cta_json_dict['patientID'] + '_' + cta_json_dict['studyUID']
#        if psid not in cta_psid_json_dicts:
#            cta_psid_json_dicts[psid] = cta_json_dict
#        else:
#            cta_psid_json_dicts[psid]['nodes'] += cta_json_dict['nodes']
#    return cta_psid_json_dicts


def gen_cta_psid_json_dict(cta_json_dict_list):
    cta_psid_json_dicts = defaultdict(list)
    for cta_json_dict in cta_json_dict_list:
        if cta_json_dict['other_info']['passReason'] != '':
            continue
        if len(cta_json_dict['nodes']) == 0:
            continue
        psid = cta_json_dict['patientID'] + '_' + cta_json_dict['studyUID']
        cta_psid_json_dicts[psid].append(cta_json_dict)
#        if psid not in cta_psid_json_dicts:
#            cta_psid_json_dicts[psid] = cta_json_dict
#        else:
#            cta_psid_json_dicts[psid]['nodes'] += cta_json_dict['nodes']
    return cta_psid_json_dicts

cta_json_dict_list = json_load('/mnt/users/code/cta/task_1985_0-1613536411.json') #/data1/wangsiwen/task_1739_0-1603443934.json
cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list)
for psid in sorted(cta_psid_json_dicts.keys()):
    if not psid.split('_')[0] == '1004596':
        continue
    cta_json_dict = cta_psid_json_dicts[psid]
    for n in cta_json_dict:
        for k, v in n.items():
            if k != 'nodes':
                print(k, v)
