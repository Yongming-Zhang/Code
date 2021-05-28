# -*- coding: utf-8 -*-
import cv2
import struct
from cta_util import *
from scipy import ndimage
from lib.utils.contour import find_contours, fill_contours, draw_contours
from scipy.interpolate import griddata
import time

def compute_dice(pred, gt): 
    pred, gt = pred.flatten(), gt.flatten()
    num = numpy.sum(pred * gt)
    den1 = numpy.sum(pred)
    den2 = numpy.sum(gt)
    epsilon = 1e-4
    dice = (2 * num + epsilon) / (den1 + den2 + epsilon)
    return dice

def load_cpr_coord_map(file_path):
    f = open(file_path, mode='rb')
    a = f.read()
    w, h, c, t = struct.unpack('iiii', a[:16])
#    x, y, z, c = struct.unpack('iiii', a[:16])
#    print(x,y,z,c)
    assert c == 3 and t == 10, 'The third and fourth items of cpr coor map should be 3 and 10'
#    assert c == 2 and t == 10
    maps = struct.unpack('f' * w * h * c, a[16:])
    maps = numpy.float32(maps).reshape(h, w, c)

 #   maps = struct.unpack('f' * x * y * z * c, a[20:])
#    maps = numpy.float32(maps).reshape(z, y, x, c)

  #  print(maps)
    return maps


class CtaCprMapper(object):
    def __init__(self, cpr_coord_maps):
        self.cpr_coord_maps = cpr_coord_maps

    def cta2cpr(self, cta_data, mode='nearest'):
        # assert mode == 'nearest', 'Only support nearest interplotation'
        cta_d, cta_h, cta_w = cta_data.shape[:3]
        cpr_data_list = []
        for cpr_coord_map in self.cpr_coord_maps:
            cpr_h, cpr_w = cpr_coord_map.shape[:2]
            if mode == 'nearest':
                cpr_data = numpy.ones([cpr_h, cpr_w] + list(cta_data.shape[3:]), cta_data.dtype) * -1024
                for i in range(cpr_h):
                    for j in range(cpr_w):
                        x, y, z = cpr_coord_map[i, j]
                        x, y, z = int(round(x)), int(round(y)), int(round(z))
                        if 0 <= x < cta_w and 0 <= y < cta_h and 0 <= z < cta_d:
                            cpr_data[i, j] = cta_data[z, y, x]
            else:
                xs, ys, zs = cpr_coord_map[:, :, 0], cpr_coord_map[:, :, 1], cpr_coord_map[:, :, 2]
                cpr_data = ndimage.map_coordinates(cta_data, [zs, ys, xs], order=1, cval=-1024).astype(cta_data.dtype)
            cpr_data_list.append(cpr_data)
        return cpr_data_list
  
    def cpr2cta_reverse_map_std(self, cpr_data_list, cta_shape, roi_index_list):
        cta_data_reverse_grid = []
        cta_data_reverse_value = []
        axis_3d=[]
        cta_data_reverse = numpy.zeros(cta_shape, dtype='uint8')
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        time1 = time.time()
        for cpr_coord_map, cpr_data in zip([self.cpr_coord_maps[i] for i in roi_index_list], cpr_data_list):
            cpr_h, cpr_w = cpr_coord_map.shape[:2]
            if not cpr_data.any() > 0:
                continue
            contours = find_contours(cpr_data)
            for k in range(len(contours)):
                xmin, xmax = numpy.array(contours[k])[:,0,0].min(), numpy.array(contours[k])[:,0,0].max()
                ymin, ymax = numpy.array(contours[k])[:,0,1].min(), numpy.array(contours[k])[:,0,1].max()
                cta_data_reverse_grid_per_contour = []
                cta_data_reverse_value_per_contour = []

                for i in range(max(0,ymin-2), min(ymax+3,cpr_h)):
                    for j in range(max(0,xmin-2), min(xmax+3,cpr_w)):
                        x, y, z = cpr_coord_map[i, j]
                        v = cpr_data[i, j]
                        cta_data_reverse_grid.append([z, y, x])
                        cta_data_reverse_value.append(v)              
                        cta_data_reverse_grid_per_contour.append([z, y, x])
                        cta_data_reverse_value_per_contour.append(v)                
                zmin_3d, zmax_3d = numpy.min(numpy.array(cta_data_reverse_grid_per_contour)[:, 0]),numpy.max(numpy.array(cta_data_reverse_grid_per_contour)[:, 0])
                ymin_3d, ymax_3d = numpy.min(numpy.array(cta_data_reverse_grid_per_contour)[:, 1]),numpy.max(numpy.array(cta_data_reverse_grid_per_contour)[:, 1])                 
                xmin_3d, xmax_3d = numpy.min(numpy.array(cta_data_reverse_grid_per_contour)[:, 2]),numpy.max(numpy.array(cta_data_reverse_grid_per_contour)[:, 2])

                for z in range(int(zmin_3d), int(zmax_3d)+1):
                    for y in range(int(ymin_3d), int(ymax_3d)+1):
                        for x in range(int(xmin_3d), int(xmax_3d)+1):
                            axis_3d.append([z, y, x])
                
#                for z in range(int(zmin_3d), int(zmax_3d)+1):
#                    for y in range(int(ymin_3d), int(ymax_3d)+1):
#                        for x in range(int(xmin_3d), int(xmax_3d)+1):
#                            dis = numpy.linalg.norm(numpy.array((z,y,x))-numpy.array(cta_data_reverse_grid_per_contour),axis=-1)
#                            indice = dis.argmin(axis=0)
#                            cta_data_reverse[z,y,x] = cta_data_reverse_value_per_contour[indice]
        

#        axis_3d = numpy.unique(numpy.array(axis_3d),axis=0)
#        cta_data_reverse_grid = numpy.array(cta_data_reverse_grid)
#        #axis_3d[:,None,0] - cta_data_reverse_grid[None,:,0]
#        for z, y, x in numpy.unique(numpy.array(axis_3d),axis=0):
#            time_sub_0 = time.time()
#            dis = numpy.linalg.norm(numpy.array((z,y,x)) - numpy.array(cta_data_reverse_grid), axis=-1)
#            time_sub_1 = time.time()
#            indice = dis.argmin(axis=0)
#            cta_data_reverse[z,y,x] = cta_data_reverse_value[indice]
#            time_sub_2 = time.time()



        if cta_data_reverse_value == []:
              return numpy.zeros(cta_shape, dtype='uint8')            
        grid_z, grid_y, grid_x = numpy.mgrid[0: cta_shape[0], 0: cta_shape[1], 0: cta_shape[2]]
        print(len(cta_data_reverse_value))
        cta_data_reverse = griddata(numpy.array(cta_data_reverse_grid), numpy.array(cta_data_reverse_value), (grid_z, grid_y, grid_x), method='linear', fill_value=0).astype(cpr_data_list[0].dtype)
        time2 = time.time()
        print(time2-time1)   
        return cta_data_reverse


    def cpr2cta_reverse_map_b2(self, cpr_data_list, cta_shape, cta_mask):
        cta_data_reverse_grid = []
        cta_data_reverse_value = []
        axis_3d=[]
        cta_data_reverse = numpy.zeros(cta_shape, dtype='uint8')
        time1 = time.time()
        for cpr_coord_map, cpr_data in zip(self.cpr_coord_maps, cpr_data_list):
            cpr_h, cpr_w = cpr_coord_map.shape[:2]
            if not cpr_data.any() > 0:
                continue
            contours = find_contours(cpr_data)
            for k in range(len(contours)):
                xmin, xmax = numpy.array(contours[k])[:,0,0].min(), numpy.array(contours[k])[:,0,0].max()
                ymin, ymax = numpy.array(contours[k])[:,0,1].min(), numpy.array(contours[k])[:,0,1].max()
                cta_data_reverse_grid_per_contour = []
                cta_data_reverse_value_per_contour = []
                           
                for i in range(max(0,ymin-2), min(ymax+3,cpr_h)):
                    for j in range(max(0,xmin-2), min(xmax+3,cpr_w)): 
                        x, y, z = cpr_coord_map[i, j]
                        v = cpr_data[i, j]
                        cta_data_reverse_grid.append([z, y, x])
                        cta_data_reverse_value.append(v)
                        cta_data_reverse_grid_per_contour.append([z, y, x])
                        cta_data_reverse_value_per_contour.append(v)
                zmin_3d, zmax_3d = numpy.min(numpy.array(cta_data_reverse_grid_per_contour)[:, 0]),numpy.max(numpy.array(cta_data_reverse_grid_per_contour)[:, 0])
                ymin_3d, ymax_3d = numpy.min(numpy.array(cta_data_reverse_grid_per_contour)[:, 1]),numpy.max(numpy.array(cta_data_reverse_grid_per_contour)[:, 1])
                xmin_3d, xmax_3d = numpy.min(numpy.array(cta_data_reverse_grid_per_contour)[:, 2]),numpy.max(numpy.array(cta_data_reverse_grid_per_contour)[:, 2])

                for z in range(int(zmin_3d), int(zmax_3d)+1):
                    for y in range(int(ymin_3d), int(ymax_3d)+1):
                        for x in range(int(xmin_3d), int(xmax_3d)+1):
                            axis_3d.append([z, y, x])

#                for z in range(int(zmin_3d), int(zmax_3d)+1):
#                    for y in range(int(ymin_3d), int(ymax_3d)+1):
#                        for x in range(int(xmin_3d), int(xmax_3d)+1):
#                            dis = numpy.linalg.norm(numpy.array((z,y,x))-numpy.array(cta_data_reverse_grid_per_contour),axis=-1)
#                            indice = dis.argmin(axis=0)
#                            cta_data_reverse[z,y,x] = cta_data_reverse_value_per_contour[indice]



#        if cta_data_reverse_value == []:
#              return numpy.zeros(cta_shape, dtype='uint8')
#        axis_3d = numpy.unique(numpy.array(axis_3d),axis=0)
#        cta_data_reverse_grid = numpy.array(cta_data_reverse_grid)
#        #axis_3d[:,None,0] - cta_data_reverse_grid[None,:,0]
#        for z, y, x in numpy.unique(numpy.array(axis_3d),axis=0):
#            time_sub_0 = time.time()
#            dis = numpy.linalg.norm(numpy.array((z,y,x)) - numpy.array(cta_data_reverse_grid), axis=-1)
#            time_sub_1 = time.time()
#            indice = dis.argmin(axis=0)
#            cta_data_reverse[z,y,x] = cta_data_reverse_value[indice]
#            time_sub_2 = time.time()




        if cta_data_reverse_value == []:
              return numpy.zeros(cta_shape, dtype='uint8')
        grid_z, grid_y, grid_x = numpy.mgrid[0: cta_shape[0], 0: cta_shape[1], 0: cta_shape[2]]
 #       print(len(cta_data_reverse_value))
        cta_data_reverse = griddata(numpy.array(cta_data_reverse_grid), numpy.array(cta_data_reverse_value), (grid_z, grid_y, grid_x), method='nearest', fill_value=0).astype(cta_mask.dtype)
        time2 = time.time() 
#        print(time2-time1)
        return cta_data_reverse

def map_cta_ann_json_to_cpr(cpr_coord_maps, cta_ann_json_dict, cta_shape, cta_instance_dict, cpr_instances):
    mapper = CtaCprMapper(cpr_coord_maps)
    cpr_json_dict = {
        'patientID': cta_ann_json_dict[0]['patientID'],
        'studyUID': cta_ann_json_dict[0]['studyUID'],
        'other_info': {'doctorId': cta_ann_json_dict[0]['other_info']['doctorId']},
        'nodes': []
    }
    cta_data_reverse_all = numpy.zeros(cta_shape, dtype='uint8')
    cta_data_origin_all = numpy.zeros(cta_shape, dtype='uint8')
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cta_mask = numpy.zeros(cta_shape, dtype='uint8')  
    cpr_mask_list = []
    roi_index_list = []
    for n in cta_ann_json_dict:
#        print(n['patientID'], n['studyUID'], n['seriesUID'])
        for node in n['nodes']:
            if isinstance(node['descText'][1][0][0]['select'], list):
                continue
            les_type = g_ccta_lesion_type_dict[node['descText'][1][0][0]['select']]
            stenosis = g_stenosis_type_dict[node['descText'][1][1][0]['select']]
            detect_necessity = g_detect_necessity_dict[node['descText'][1][2][0]['select']]
            segments = ['other' if s == u'其他' else s for s in node['descText'][1][3][0]['select']]
            if les_type == 'none':
                continue
            new_node = {'type': les_type, 'detect_necessity': detect_necessity, 'segments': segments}
            if les_type in ['stent_clear', 'stent_unclear', 'stent_block', 'stent_none', 'mca', 'mca_mb']:
                pts3d = []
#                for roi in node['rois']:
#                    z = cta_instance_dict[roi['slice_index']]
#                    x, y = roi['edge'][0]
#                    pts3d.append([x, y, z])
#                new_node['pts3d'] = pts3d
            else:
                new_node['stenosis'] = stenosis
#                cta_mask = numpy.zeros(cta_shape, dtype='uint8')
#                cpr_mask_list = []
#                roi_index_list = []
                for roi in node['rois'] + node['bounds']:




                    contours_numpy = numpy.int32(roi['edge'])
                    z = cta_instance_dict[roi['slice_index']]
                    cv2.drawContours(cta_mask[z], (contours_numpy,), 0, color=255, thickness=-1)
#                cpr_masks = mapper.cta2cpr(cta_mask, 'trilinear')
#                cta_data_reverse = mapper.cpr2cta_reverse_map_b2(cpr_masks, cta_shape, cta_mask)
#                cta_data_origin_all += cta_mask
#                cta_data_reverse_all += cta_data_reverse
    cpr_masks = mapper.cta2cpr(cta_mask, 'trilinear')
    cta_data_reverse_all = mapper.cpr2cta_reverse_map_b2(cpr_masks, cta_shape, cta_mask)
    #dice = compute_dice(cta_data_reverse_all, cta_mask)
    #print(cta_data_reverse_all[cta_data_reverse_all>0],cta_mask[cta_mask>0])
    cta_data_reverse_all[cta_data_reverse_all>0] = 1
    for i in range(cta_data_reverse_all.shape[0]):
        cta_data_reverse_all[i] = cv2.erode(cta_data_reverse_all[i], kernel_erode)
    cta_mask[cta_mask>0] = 1
    dice = compute_dice(cta_data_reverse_all, cta_mask)
    print(dice)

                

#                    if roi['slice_index'] < 0:
#                        continue
#                    contours_numpy = numpy.int32(roi['edge'])
##                    print(n['patientID'], n['seriesUID'] + '_' + str(roi['slice_index']), cpr_instances.count(n['seriesUID'] + '_' + str(roi['slice_index']))) 
#                    assert cpr_instances.count(n['seriesUID'] + '_' + str(roi['slice_index'])) == 1
#                    roi_index = cpr_instances.index(n['seriesUID'] + '_' + str(roi['slice_index']))
#                    cpr_mask = numpy.zeros(cpr_coord_maps[roi_index].shape[:2], dtype='uint8')
#                    cv2.drawContours(cpr_mask, (contours_numpy,), 0, color=1, thickness=-1)
#                    
#                    cpr_mask_list.append(cpr_mask)
#                    roi_index_list.append(roi_index)
##                cta_data_reverse = mapper.cpr2cta_reverse_map_std(cpr_mask_list, cta_shape, roi_index_list)
##                cta_data_reverse_all += cta_data_reverse 
#    cta_data_reverse = mapper.cpr2cta_reverse_map_std(cpr_mask_list, cta_shape, roi_index_list)
#    cta_data_reverse_all += cta_data_reverse


    return cta_data_reverse_all, dice
#                    z = cta_instance_dict[roi['slice_index']]
#                    cv2.drawContours(cta_mask[z], (contours_numpy,), 0, color=255, thickness=-1)
#                cpr_masks = mapper.cta2cpr(cta_mask, 'trilinear')
#                cta_data_reverse = mapper.cpr2cta_reverse_map(cpr_masks, cta_shape, cta_mask)
#                rois = []
#                for i, cpr_mask in enumerate(cpr_masks):
#                    edge_list = find_contours(cpr_mask)
#                    for edge in edge_list:
#                        rois.append({'slice_index': cpr_instances[i], 'edge': edge[:, 0, :].tolist()})
#                if len(rois) == 0:
#                    new_node = None
#                else:
#                    new_node['rois'] = rois
#            if new_node is not None:
#                cpr_json_dict['nodes'].append(new_node)
#    return cpr_json_dict



#if __name__ == '__main__':
#    load_cpr_coord_map('/data1/wangsiwen//coro001_0.dat')
#    load_cpr_coord_map('/data1/inputdata/0657758/089FA286/95BAD70B_CTA/cprCoronary/coro004_162.dat')
