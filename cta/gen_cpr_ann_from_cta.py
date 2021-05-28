from cta_cpr_map import *
from lib.utils.mio import *
import glob
import numpy
from lib.image.image import gray2rgb
from lib.image.draw import draw_texts
import pydicom
from collections import defaultdict
import time
import dicom2nifti
import SimpleITK as sitk


def dcm2nii(path_read, path_save):
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    print(len(series_file_names))
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, path_save)

def draw_save_cpr_one(cpr_json_dict, dcm_paths, cpr_instances, save_dir, is_draw_text=True):
    mkdir_safe(save_dir)
    images = [gen_image_from_dcm(dcm_path, [(-100, 900)])[0] for dcm_path in dcm_paths]
    draw_images = [gray2rgb(img) for img in images]
    color_dict = {'cal': (0, 255, 0), 'low': (0, 0, 255), 'mix': (0, 97, 255), 'block': (255, 0, 255)}
    instance_index_dict = {instance: i for i, instance in enumerate(cpr_instances)}
    for node in cpr_json_dict['nodes']:
        if node['type'] not in color_dict:
            continue
        color = color_dict[node['type']]
        for roi in node['rois']:
            edge_numpy = numpy.int32(roi['edge'])
            draw_img = draw_images[instance_index_dict[roi['slice_index']]]
            fill_contours(draw_img, edge_numpy, color)
            x0, y0 = edge_numpy.min(axis=0)
            if is_draw_text:
                draw_texts(draw_img, [x0, y0], [node['stenosis']], color, font_scale=0.5)
    for img, draw_img, dcm_path in zip(images, draw_images, dcm_paths):
        save_path = save_dir + dcm_path.split('/')[-1][:-4] + '.png'
        save_image(save_path, numpy.hstack([gray2rgb(img), draw_img]))

def draw_save_cta(psid, cta_data_reverse_all, dcm_paths):
    save_dir = os.path.join('/data1/wangsiwen/cta_save_png_std_3d/', psid.split('_')[0] + '/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    images = [gen_image_from_dcm(dcm_path, [(-100, 900)])[0] for dcm_path in dcm_paths]
    #print(dcm_paths)
    #dicom2nifti.dicom_series_to_nifti('/'.join(dcm_paths[0].split('/')[:-1]), save_dir, reorient_nifti=True)
    print('/'.join(dcm_paths[0].split('/')[:-1]))
    dcm2nii('/'.join(dcm_paths[0].split('/')[:-1]), save_dir)


#    draw_images = [gray2rgb(img) for img in images]
#    color_dict = {'cal': (0, 255, 0), 'low': (0, 0, 255), 'mix': (0, 97, 255), 'block': (255, 0, 255)}
#    color = color_dict['cal']
#    for i in range(cta_data_reverse_all.shape[0]):
#        rois = []
#        edge_list = find_contours(cta_data_reverse_all[i])
#        for edge in edge_list:
#            rois.append(edge[:, 0, :].tolist())
#        draw_img = draw_images[i]
#        for roi in rois:
#            edge_numpy = numpy.int32(roi)
#            fill_contours(draw_img, edge_numpy, color)
#    for img, draw_img, dcm_path in zip(images, draw_images, dcm_paths):
#        save_path = save_dir + dcm_path.split('/')[-1][:-4] + '.png'
#        save_image(save_path, numpy.hstack([gray2rgb(img), draw_img]))



def gen_cta_psid_json_dict(cta_json_dict_list):
    cta_psid_json_dicts = defaultdict(list)
    for cta_json_dict in cta_json_dict_list:
        if cta_json_dict['other_info']['passReason'] != '':
            continue
        if len(cta_json_dict['nodes']) == 0:
            continue
#        if 'isReview' in cta_json_dict['other_info'].keys() and cta_json_dict['other_info']['isReview'] == 0:
#            continue
        psid = cta_json_dict['patientID'] + '_' + cta_json_dict['studyUID']
        cta_psid_json_dicts[psid].append(cta_json_dict)
#        if psid not in cta_psid_json_dicts:
#            cta_psid_json_dicts[psid] = cta_json_dict
#        else:
#            cta_psid_json_dicts[psid]['nodes'] += cta_json_dict['nodes']
    return cta_psid_json_dicts


def has_empty_node(cta_json_dict):
    #for node in cta_json_dict['nodes']:
    for n in cta_json_dict:
        for node in n['nodes']:
            if isinstance(node['descText'][1][0][0]['select'], list):
                continue
            if len(node['descText'][1][0][0]['select']) == 0:
                return True
    return False


def gen_cpr_json_main():
    # psid_list = [
    #     '0091321_1.2.392.200036.9116.2.5.1.37.2417525831.1552887510.650964',
    #     '0055788_1.2.392.200036.9116.2.5.1.37.2417525831.1551677413.80373'
    # ]
    base_dir = '/data2/zhangfd/data/cta/b2_407/cta2cpr/'
    cpr_dir = base_dir + 'cpr_s9_n20_reorg/'
    # cpr_dir = base_dir + 'cpr_lumen_dat_s9_n20/'
    cta_dcm_dir = base_dir + 'cta_dicom/'


#    base_dir = '/data2/zhangfd/data/cta/std_test/b1_150/'
#    cpr_dir = base_dir + 'cpr_lumen_s18_n20_raw/'
#    cta_dcm_dir = base_dir + 'dicom/'


    cta_ann_json_dir = base_dir + 'annotation/'
    cpr_json_dir = cta_ann_json_dir + 'cpr_s9_n20_reorg/check_json/'
    draw_save_dir = cta_ann_json_dir + 'cpr_s9_n20_reorg/draw_check/'
    # cpr_json_dir = cta_ann_json_dir + 'lumen_s9_n20/raw_json/'
    # draw_save_dir = cta_ann_json_dir + 'lumen_s9_n20/draw_raw/'
#    mkdir_safe(cpr_json_dir)
    # cta_json_dict_list = json_load(cta_ann_json_dir + 'task_1265_0-1588272710.json')
    cta_json_dict_list = json_load('/mnt/users/code/cta/task_1985_0-1613536411.json')
    #cta_json_dict_list += json_load('/data1/wangsiwen/task_1772_0-1602303384.json')


#    cta_json_dict_list = json_load('/data1/wangsiwen/task_1739_0-1603443934.json')


    cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list)
    is_lumen = False
    vessel_name_map = load_vessel_name_map()
    print('%d psid' % len(cta_psid_json_dicts))
    avg_dice = []
    avg_dice_except_neg = []
    for psid in sorted(cta_psid_json_dicts.keys()):
        # if psid.split('_')[0] < '0784330':
        #     continue
        cta_json_dict = cta_psid_json_dicts[psid]
#        psid = cta_json_dict['patientID'] + '_' + cta_json_dict['studyUID']
        if has_empty_node(cta_json_dict):
            print('%s has empty nodes, skip' % psid)
            continue
        # if cta_json_dict['patientID'] <= '0658008':
        #     continue
        # if cta_json_dict['patientID'] <= '0784420':
        #     continue
        # if cta_json_dict['patientID'] <= '0784476':
        #     continue
        # if psid not in psid_list:
        #     continue
        if is_lumen:
            cpr_series_dirs = sorted(glob.glob(cpr_dir + psid + '/lumen/*/'))
        else:
            cpr_series_dirs = sorted(glob.glob(cpr_dir + psid + '/*/'))
#            cpr_series_dirs = sorted(glob.glob(cpr_dir + psid + '*' + '/cpr/'))
        
        cta_dcm_paths = sorted(glob.glob(cta_dcm_dir + psid + '/*.dcm'))
#        cta_dcm_paths = sorted(glob.glob(cta_dcm_dir + psid + '*' + '/*.dcm'))
        cta_instance_dict = {int(p.split('/')[-1][:-4]): i for i, p in enumerate(cta_dcm_paths)}
        cta_shape = [len(cta_instance_dict), 512, 512]

        cpr_instances = []
        cpr_coord_maps = []
        time1=time.time()
        for cpr_series_dir in cpr_series_dirs:
            if cpr_series_dir == 'centerline':
                continue
            vessel_name = cpr_series_dir.split('/')[-2]
            if vessel_name == 'centerline':
                continue
            vessel_name = vessel_name_map.get(vessel_name, vessel_name)
            cpr_dcm_paths = sorted(glob.glob(cpr_series_dir + '*.dcm'))
            cpr_instances += [dcm_path.split('/')[-1].split('_')[0] + '_' + str(int(dicom_load(dcm_path).InstanceNumber)) for dcm_path in cpr_dcm_paths]
            cpr_coord_maps += [load_cpr_coord_map(cpr_dcm_path + '.dat') for cpr_dcm_path in cpr_dcm_paths]

        cta_data_reverse_all, dice = map_cta_ann_json_to_cpr(
                cpr_coord_maps, cta_json_dict, cta_shape, cta_instance_dict, cpr_instances)
        time2=time.time()
        avg_dice.append(dice)
        if dice < 1.0:
            avg_dice_except_neg.append(dice)
        print(psid,sum(avg_dice) / (len(avg_dice)+1e-5), sum(avg_dice_except_neg) / (len(avg_dice_except_neg)+1e-5))

#        print(time2-time1)
#        print(psid, cta_data_reverse_all[cta_data_reverse_all>1])                
#        draw_save_cta(psid, cta_data_reverse_all, cta_dcm_paths)
                           
#        cpr_json_dict = map_cta_ann_json_to_cpr(
#                cpr_coord_maps, cta_json_dict, cta_shape, cta_instance_dict, cpr_instances)
#        print('%s/%s: %d cta nodes, %d cpr nodes' % (
#                psid, vessel_name, len(cta_json_dict['nodes']), len(cpr_json_dict['nodes'])))
#            json_save(cpr_json_dir + psid + '_' + vessel_name + '.json', cpr_json_dict)
#            if len(cpr_json_dict['nodes']) > 0:
#                cur_draw_dir = draw_save_dir + psid + '_' + vessel_name + '/'
#                draw_save_cpr_one(cpr_json_dict, cpr_dcm_paths, cpr_instances, cur_draw_dir, not is_lumen)


if __name__ == '__main__':
    gen_cpr_json_main()

