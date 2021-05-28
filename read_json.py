import json

cta_json_src_dict = '/mnt/users/code/cta/task_1985_0-1613536411.json'
with open(cta_json_src_dict, 'r') as f:
    datas = json.load(f)
    #datas = f.read()
    series = []
    for data in datas:
        if data['seriesUID'] not in series and '.' not in data['seriesUID']:
            print(data['seriesUID'])
            series.append(data['seriesUID'])
    print(series)