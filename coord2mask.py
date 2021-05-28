import json
import numpy

def json_to_cpr_mask():
    cta_json_dict = '/mnt/users/code/cta/task_1985_0-1613536411.json'
    #cpr_mask_dict = ''
    with open(cta_json_dict, 'r') as f:
        datas = json.load(f)
        for data in datas:
            print(data['patientID'])
            print(data['studyUID'])
            print(data['seriesUID'])
            nodes = data['nodes']
            for node in nodes:
                rois = node['rois']
                for roi in rois:
                    #print(roi['edge'])
                    print()
                    #for 



if __name__ == '__main__':
    json_to_cpr_mask()