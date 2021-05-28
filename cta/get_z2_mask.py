# -*- coding: utf-8 -*-
from PIL import Image
import cv2
import struct
from cta_util import *
from scipy import ndimage
from lib.utils.contour import find_contours, fill_contours, draw_contours
from scipy.interpolate import griddata
import time
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
import numpy as np
from itertools import chain
import math

def get_tangent_direction(c2d_pts, idx):
    # compute normal direction
    i0 = i1 = idx
    stride, thresh = 10, 0.5
    cx, cy = c2d_pts[idx]
    for i in range(idx - stride, 0, -stride):
        x, y = c2d_pts[i]
        if abs(cx - x) > thresh and abs(cy - y) > thresh:
            i0 = i
            break
    for i in range(idx + stride, len(c2d_pts), stride):
        x, y = c2d_pts[i]
        if abs(cx - x) > thresh and abs(cy - y) > thresh:
            i1 = i
            break
    return c2d_pts[i1][0] - c2d_pts[i0][0], c2d_pts[i1][1] - c2d_pts[i0][1]


def get_normal_diameter(mask, cx, cy, dx, dy):
    # compute vessel diameter
    nx, ny = -dy, dx
    cx, cy = int(round(cx)), int(round(cy))
    if abs(dx) + abs(dy) < 0.001:
        return cx, cy, 0, cx, cy, 0, 0
    h, w = mask.shape
    cx = min(max(0, cx), w - 1)
    cy = min(max(0, cy), h - 1)
    x0, y0, x1, y1 = -1, -1, -1, -1
    x, y = cx, cy
    off0, off1 = 0, 0

    points_list = []
    # find the first non-zero point as (cx, cy), ugly coding due to emergency -_-!
    if mask[int(round(cy)), int(round(cx))] == 0:
        offsets = [0]
        for offx in range(1, 20):
            offsets += [offx, -offx]
        if abs(nx) > abs(ny):
            # along x-axis, right and left
            for offx in offsets:
                y = ny / nx * offx + cy
                y_int, x_int = int(round(y)), int(round(cx + offx))
                if h > y_int >= 0 and w > x_int >= 0 and mask[y_int, x_int] > 0:
                    cx, cy = cx + offx, y
                    break
        else:
            # along y-axis, up and down
            for offy in offsets:
                x = nx / ny * offy + cx
                y_int, x_int = int(round(cy + offy)), int(round(x))
                if h > y_int >= 0 and w > x_int >= 0 and mask[y_int, x_int] > 0:
                    cx, cy = x, cy + offy
                    break
    if mask[int(round(cy)), int(round(cx))] == 0:
        return []
    # compute diameter
    if abs(nx) > abs(ny):
        # along x-axis, right and left
        cnt_0 = 0
        for x in range(cx + 1, w):
            y = ny / nx * (x - cx) + cy
            y_int = int(round(y))
            off1 = x - cx
            points_list.append([x, y_int])
            if y_int >= h or y_int < 0 or mask[y_int, x] == 0:
                if cnt_0 > 2:
                    break
                cnt_0 += 1
        cnt_0 = 0
        for x in range(cx - 1, 0, -1):
            y = ny / nx * (x - cx) + cy
            y_int = int(round(y))
            off0 = x - cx
            points_list.append([x, y_int])
            if y_int >= h or y_int < 0 or mask[y_int, x] == 0:
                if cnt_0 > 2:
                    break
                cnt_0 += 1
    else:
        # along y-axis, up and down
        cnt_0 = 0
        for y in range(cy + 1, h):
            x = nx / ny * (y - cy) + cx
            x_int = int(round(x))
            off1 = y - cy
            points_list.append([x_int, y])
            if x_int >= w or x_int < 0 or mask[y, x_int] == 0:
                if cnt_0 > 2:
                    break
                cnt_0 += 1
        cnt_0 = 0
        for y in range(cy - 1, 0, -1):
            x = nx / ny * (y - cy) + cx
            x_int = int(round(x))
            off0 = y - cy
            points_list.append([x_int, y])
            if x_int >= w or x_int < 0 or mask[y, x_int] == 0:
                if cnt_0 > 2:
                    break
                cnt_0 += 1
    return points_list


def get_normal_diameter_xy(mask, cx, cy, dx, dy):
    # compute vessel diameter
    nx, ny = -dy, dx
    cx, cy = int(round(cx)), int(round(cy))
    if abs(dx) + abs(dy) < 0.001:
        return cx, cy, 0, cx, cy, 0, 0
    h, w = mask.shape
    cx = min(max(0, cx), w - 1)
    cy = min(max(0, cy), h - 1)
    x0, y0, x1, y1 = -1, -1, -1, -1
    x, y = cx, cy
    off0, off1 = 0, 0

    points_list = []
    # find the first non-zero point as (cx, cy), ugly coding due to emergency -_-!
    if mask[int(round(cy)), int(round(cx))] == 0:
        offsets = [0]
        for offx in range(1, 20):
            offsets += [offx, -offx]
        if abs(nx) > abs(ny):
            # along x-axis, right and left
            for offx in offsets:
                y = ny / nx * offx + cy
                y_int, x_int = int(round(y)), int(round(cx + offx))
                if h > y_int >= 0 and w > x_int >= 0 and mask[y_int, x_int] > 0:
                    cx, cy = cx + offx, y
                    break
        else:
            # along y-axis, up and down
            for offy in offsets:
                x = nx / ny * offy + cx
                y_int, x_int = int(round(cy + offy)), int(round(x))
                if h > y_int >= 0 and w > x_int >= 0 and mask[y_int, x_int] > 0:
                    cx, cy = x, cy + offy
                    break
    if mask[int(round(cy)), int(round(cx))] == 0:
        return []
    # compute diameter
    if abs(nx) > abs(ny):
        # along x-axis, right and left
        cnt_0 = 0
        for x in range(cx + 1, w):
            y = cy
            y_int = int(y)
            off1 = x - cx
            points_list.append([x, y_int])
            if y_int >= h or y_int < 0 or mask[y_int, x] == 0:
                if cnt_0 > 5:
                    break
                cnt_0 += 1
        cnt_0 = 0
        for x in range(cx - 1, 0, -1):
            y = cy
            y_int = int(y)
            off0 = x - cx
            points_list.append([x, y_int])
            if y_int >= h or y_int < 0 or mask[y_int, x] == 0:
                if cnt_0 > 5:
                    break
                cnt_0 += 1
        cnt_0 = 0
        for x in range(cx + 1, w):
            y = ny / nx * (x - cx) + cy
            y_int = int(round(y))
            off1 = x - cx
            points_list.append([x, y_int])
            if y_int >= h or y_int < 0 or mask[y_int, x] == 0:
                if cnt_0 > 5:
                    break
                cnt_0 += 1
        cnt_0 = 0
        for x in range(cx - 1, 0, -1):
            y = ny / nx * (x - cx) + cy
            y_int = int(round(y))
            off0 = x - cx
            points_list.append([x, y_int])
            if y_int >= h or y_int < 0 or mask[y_int, x] == 0:
                if cnt_0 > 5:
                    break
                cnt_0 += 1
    else:
        # along y-axis, up and down
        cnt_0 = 0
        for y in range(cy + 1, h):
            x = cx
            x_int = int(x)
            off1 = y - cy
            points_list.append([x_int, y])
            if x_int >= w or x_int < 0 or mask[y, x_int] == 0:
                if cnt_0 > 5:
                    break
                cnt_0 += 1
        cnt_0 = 0
        for y in range(cy - 1, 0, -1):
            x = cx
            x_int = int(x)
            off0 = y - cy
            points_list.append([x_int, y])
            if x_int >= w or x_int < 0 or mask[y, x_int] == 0:
                if cnt_0 > 5:
                    break
                cnt_0 += 1
        cnt_0 = 0
        for y in range(cy + 1, h):
            x = nx / ny * (y - cy) + cx
            x_int = int(round(x))
            off1 = y - cy
            points_list.append([x_int, y])
            if x_int >= w or x_int < 0 or mask[y, x_int] == 0:
                if cnt_0 > 5:
                    break
                cnt_0 += 1
        cnt_0 = 0
        for y in range(cy - 1, 0, -1):
            x = nx / ny * (y - cy) + cx
            x_int = int(round(x))
            off0 = y - cy
            points_list.append([x_int, y])
            if x_int >= w or x_int < 0 or mask[y, x_int] == 0:
                if cnt_0 > 5:
                    break
                cnt_0 += 1
    return points_list





def get_normal_diameter_by_idx(c2d_pts, idx, mask, mask_plaque=None):
    dx, dy = get_tangent_direction(c2d_pts, idx)
    cx, cy = c2d_pts[idx]
    if mask_plaque is not None:
        _, d = get_union_normal_diameter(mask, mask_plaque, cx, cy, dx, dy)
    else:
        points_list = get_normal_diameter_xy(mask, cx, cy, dx, dy)
    return points_list


def get_roi_to_cl_range(roi, cl, max_distance=10.0):
    """
    get the index range of centerline by projecting roi to its closest points
    also points with max_distance away are rejected
    Args:
        roi (list[pts])
        cl (list[pts])
        max_distance (float) : reject threshold
    Return
        min_index, max_index (int): return None if no valid points
    """
    roi = np.array(roi)
    cl = np.array(cl)
    sub = cl[None, :] - roi[:, None, ...]
    dis = np.linalg.norm(sub, axis=-1)
    valid_min_dist = dis.min(axis=1) < max_distance
    indices = dis.argmin(axis=1)[valid_min_dist]
#    print(indices.min(),indices.max())
    return indices.tolist()
 
save_gen_dir = '/data1/wangsiwen/code/cta/cta2cpr/lz/save_mask'
def draw_save_cpr(psid, dcm_paths, mask_files, mask_names, roi_index_list):
    save_dir = os.path.join(save_gen_dir, psid.split('_')[0], psid.split('_')[1], 'cpr/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    images = [gen_image_from_dcm(dcm_path, [(-100, 900)])[0] for dcm_path in dcm_paths]
    draw_images = [gray2rgb(img) for img in images]
    for draw_image, mask_file, mask_name in zip([draw_images[i] for i in roi_index_list], mask_files, mask_names):
        contours, hierarchy = cv2.findContours(mask_file, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
        draw_image_contour = cv2.drawContours(draw_image.copy(), contours, -1, (0,0,255), 1)
        #save_path_in = save_dir + mask_name + 'in_.png'
        save_path_out = save_dir + mask_name + '.png'
        #save_image(save_path_in, draw_image)
        save_image(save_path_out, draw_image_contour) 

def draw_save_cta(psid, cta_data_reverse_all, dcm_paths):
    save_dir = os.path.join(save_gen_dir, psid.split('_')[0], psid.split('_')[1] , 'axial/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    images = [gen_image_from_dcm(dcm_path, [(-100, 900)])[0] for dcm_path in dcm_paths]
    
#    #dicom2nifti.dicom_series_to_nifti('/'.join(dcm_paths[0].split('/')[:-1]), save_dir, reorient_nifti=True)
#    print('/'.join(dcm_paths[0].split('/')[:-1]))
#    dcm2nii('/'.join(dcm_paths[0].split('/')[:-1]), save_dir)

    
    out = sitk.GetImageFromArray(cta_data_reverse_all)
    sitk.WriteImage(out, os.path.join(save_dir, 'mask_plaque.nii.gz'))
    itk_img = sitk.ReadImage(os.path.join(save_dir, 'mask_plaque.nii.gz'))
    img = sitk.GetArrayFromImage(itk_img)
#    print(dcm_paths)
#    try:
#        dicom2nifti.dicom_series_to_nifti('/'.join(dcm_paths[0].split('/')[:-1]), os.path.join(save_dir, 'dcm_plaque.nii.gz'), reorient_nifti=True)
#    except:
#        print(psid, 'SAVE DCM NII ERROR')
    try:
        assert (cta_data_reverse_all == img).all()
    except:
        print(psid, 'SAVE NII ERROR')


    draw_images = [gray2rgb(img) for img in images]
    color_dict = {'cal': (0, 255, 0), 'low': (0, 0, 255), 'mix': (0, 97, 255), 'block': (255, 0, 255)}
    color = color_dict['cal']
    for i in range(cta_data_reverse_all.shape[0]):
        rois = []
        edge_list = find_contours(cta_data_reverse_all[i])
        for edge in edge_list:
            rois.append(edge[:, 0, :].tolist())
        draw_img = draw_images[i]
        for roi in rois:
            edge_numpy = numpy.int32(roi)
            fill_contours(draw_img, edge_numpy, color)
    for img, draw_img, dcm_path in zip(images, draw_images, dcm_paths):
        save_path = save_dir + dcm_path.split('/')[-1][:-4] + '.png'
        save_image(save_path, numpy.hstack([gray2rgb(img), draw_img]))


def cpr2cta_reverse_map_std(cpr_coord_maps, cpr_data_list, cta_shape, roi_index_list):
    cta_data_reverse_grid = []
    cta_data_reverse_value = []
    axis_3d=[]
    cta_data_reverse = numpy.zeros(cta_shape, dtype='uint8')
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    time1 = time.time()
    for cpr_coord_map, cpr_data in zip([cpr_coord_maps[i] for i in roi_index_list], cpr_data_list):
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

    if cta_data_reverse_value == []:
          return numpy.zeros(cta_shape, dtype='uint8')
    grid_z, grid_y, grid_x = numpy.mgrid[0: cta_shape[0], 0: cta_shape[1], 0: cta_shape[2]]
#    print(len(cta_data_reverse_value))
    cta_data_reverse = griddata(numpy.array(cta_data_reverse_grid), numpy.array(cta_data_reverse_value), (grid_z, grid_y, grid_x), method='linear', fill_value=0).astype(cpr_data_list[0].dtype)
    time2 = time.time()
    print(time2-time1)
    return cta_data_reverse                                                                                                                                                                                                      
def cpr2cta_reverse_map_std_new(cpr_coord_maps, cpr_data_list, cpr_name_list, cpr_vessel_list, cta_shape, roi_index_list, patient, series):
    cta_data_reverse_grid = []
    cta_data_reverse_value = []
    cta_data_reverse_dict = defaultdict(list)
    cta_data_reverse_grid_int = []
    cta_data_reverse_value_int = []

    axis_3d=[]
    cta_data_reverse = numpy.zeros(cta_shape, dtype='uint8')
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#    kernel_erode = np.ones((5, 5), np.uint8) 
    time1 = time.time()
    
    cl_range_list_vessel = defaultdict(list)
    len_cl_points_3d = defaultdict(list)
    cl_range_points_vessel = {}

    cl_points_list = []
    for cpr_coord_map, cpr_data, cpr_name, cpr_vessel in zip([cpr_coord_maps[i] for i in roi_index_list], cpr_data_list, cpr_name_list, cpr_vessel_list):
        cpr_h, cpr_w = cpr_coord_map.shape[:2]
        cl_path = glob.glob(os.path.join(mapping_gen_path, patient+'*'+series.split('_')[0], 'cpr', 'centerline', cpr_name.split('_')[0] + '*2d'))[0]
        with open (cl_path) as f:
            cl_files = json.load(f)
        for cl_file in cl_files['vessel_group']:
            if cpr_name == cl_file['name']:
                cl_points = cl_file['points']
                break
        len_cl_points_3d[cpr_name.split('_')[0]] = len(cl_points)
        cl_points_list.append(cl_points)
        if not cpr_data.any() > 0:
            continue
        y, x = np.where(cpr_data)
        roi = np.array([x, y]).T
        cl_range_list_vessel[cpr_name.split('_')[0]].append(get_roi_to_cl_range(roi, cl_points))
    for k, v in cl_range_list_vessel.items():
        cl_points_np = np.array(list(set(chain.from_iterable(v))))
        cl_points_more = np.array([]) 
        for i in range(-20, 20):
            cl_points_more = np.concatenate((cl_points_more, cl_points_np + i))
        cl_points_more = np.clip(np.unique(cl_points_more), 0, len_cl_points_3d[k]-1).astype(np.int16)
        cl_range_points_vessel[k] = cl_points_more
 #   print(cl_range_points_vessel)
  #  print(len_cl_points_3d)
    for cpr_coord_map, cpr_data, cpr_name, cpr_vessel, c2d_pts in zip([cpr_coord_maps[i] for i in roi_index_list], cpr_data_list, cpr_name_list, cpr_vessel_list, cl_points_list):
#        print(cpr_name)
        cpr_data = cv2.erode(cpr_data, kernel_erode)
        points_list_all = []
        cpr_h, cpr_w = cpr_coord_map.shape[:2]
        if not cpr_data.any() > 0:
            continue
#        print(cpr_name)
        for idx in cl_range_points_vessel[cpr_name.split('_')[0]]:
            points_list = get_normal_diameter_by_idx(c2d_pts, idx, cpr_vessel)
            points_list_all += points_list
#        print(cpr_name, points_list_all)
#        if points_list_all != []:
#            print(np.max(np.array(points_list_all)[...,0]), np.max(np.array(points_list_all)[...,1]))            
#            print(cpr_coord_map.shape)
        for j, i in points_list_all:
            try:
                x, y, z = cpr_coord_map[i, j]
                v = cpr_data[i, j]
                cta_data_reverse_grid.append([z, y, x])
                if cpr_vessel[i, j] == 0:
                    cta_data_reverse_value.append(v)
                else:
                    cta_data_reverse_value.append(v)
                if cpr_vessel[i, j] == 0:                    
                    cta_data_reverse_dict[(int(z), int(y), int(x))].append(v)
                else:
                    cta_data_reverse_dict[(int(z), int(y), int(x))].append(v)
            except:
                pass
                #print(i,j ,cpr_coord_map.shape)
        
#    zmin_3d, zmax_3d = numpy.min(numpy.array(cta_data_reverse_grid_per_contour)[:, 0]),numpy.max(numpy.array(cta_data_reverse_grid_per_contour)[:, 0])
#    ymin_3d, ymax_3d = numpy.min(numpy.array(cta_data_reverse_grid_per_contour)[:, 1]),numpy.max(numpy.array(cta_data_reverse_grid_per_contour)[:, 1])
#    xmin_3d, xmax_3d = numpy.min(numpy.array(cta_data_reverse_grid_per_contour)[:, 2]),numpy.max(numpy.array(cta_data_reverse_grid_per_contour)[:, 2])
#  
#    for z in range(int(zmin_3d), int(zmax_3d)+1):
#        for y in range(int(ymin_3d), int(ymax_3d)+1):
#            for x in range(int(xmin_3d), int(xmax_3d)+1):
#                axis_3d.append([z, y, x])

    if cta_data_reverse_value == []:
          return numpy.zeros(cta_shape, dtype='uint8')
    for k, v in cta_data_reverse_dict.items():
        cta_data_reverse_grid_int.append(list(k))
        if v.count(255) >= v.count(0):
            cta_data_reverse_value_int.append(255)
        else:
            cta_data_reverse_value_int.append(0)
    grid_z, grid_y, grid_x = numpy.mgrid[0: cta_shape[0], 0: cta_shape[1], 0: cta_shape[2]]
    print(len(cta_data_reverse_value), len(cta_data_reverse_value_int))

#    cta_data_reverse = griddata(numpy.array(list(cta_data_reverse_dict_int.keys())), numpy.array(list(cta_data_reverse_dict_int.values())), (grid_z, grid_y, grid_x), method='linear', fill_value=0).astype(cpr_data_list[0].dtype)
#    cta_data_reverse = griddata(numpy.array(cta_data_reverse_grid_int), numpy.array(cta_data_reverse_value_int), (grid_z, grid_y, grid_x), method='linear', fill_value=0).astype(cpr_data_list[0].dtype)
    final_dict = defaultdict(list)
    final_dict_one = {}
    for (z,y,x), gray in zip(cta_data_reverse_grid, cta_data_reverse_value):
#        final_dict[int(round(z)), int(round(y)), int(round(x))].append(gray)
        final_dict[int(math.ceil(z)), int(math.ceil(y)), int(math.ceil(x))].append(gray)
        final_dict[int(math.ceil(z)), int(math.ceil(y)), int(x)].append(gray)
        final_dict[int(math.ceil(z)), int(y), int(math.ceil(x))].append(gray)
        final_dict[int(math.ceil(z)), int(y), int(x)].append(gray)

        final_dict[int(z), int(math.ceil(y)), int(math.ceil(x))].append(gray)
        final_dict[int(z), int(math.ceil(y)), int(x)].append(gray)
        final_dict[int(z), int(y), int(math.ceil(x))].append(gray)
        final_dict[int(z), int(y), int(x)].append(gray)

    for k,v in final_dict.items():
#       print(v.count(255), v.count(0)) 
       if v.count(255) >= 8:
            cta_data_reverse[k[0], k[1], k[2]] = 1
    
    #for (z,y,x), gray in zip(cta_data_reverse_grid, cta_data_reverse_value):
    #    cta_data_reverse[int(round(z)), int(round(y)), int(round(x))] += gray
    time2 = time.time()
    print(time2-time1)
    return cta_data_reverse                     

#plaque_mask_gen_path = '/data1/wangsiwen/plaque_mask_ffr_newpolicy/'
#cta_data_gen_path = '/data2/xiongxiaoliang/ffr_dataset/'
#mapping_gen_path = '/data1/zhangfd/data/cta/zhe2/z2/cpr_lumen_s18_n10/'


plaque_mask_gen_path = '/data1/wangsiwen/code/cta/cta2cpr/lz/plaque_mask/'
cta_data_gen_path = '/data1/wangsiwen/code/cta/cta2cpr/lz/cta/' #/data1/zhangfd/data/cta/b4_lz493/cta_dicom
mapping_gen_path = '/data1/wangsiwen/code/cta/cta2cpr/lz/cpr_lumen_s18_n10/'


#patient_exists = os.listdir('/data1/wangsiwen/z2_save_png_newpolicy/')

for patient in sorted(os.listdir(plaque_mask_gen_path)):
#    if not patient == '1022838':
#        continue
    for series in os.listdir(os.path.join(plaque_mask_gen_path, patient)):
        print(patient, series)
        mask_paths = sorted(glob.glob(os.path.join(plaque_mask_gen_path, patient, series, '*.png')))
        mask_names = [mask_path.split('/')[-1].split('.')[0] for mask_path in mask_paths]
        mask_files = [numpy.array(Image.open(mask_path)) for mask_path in mask_paths]

        cpr_paths = glob.glob(os.path.join(mapping_gen_path, patient+'*'+series.split('_')[0], 'cpr'))[0]
        cpr_paths = sorted(glob.glob(os.path.join(cpr_paths, '*.dcm')))
        cpr_names =  [cpr_path.split('/')[-1].split('.')[0] for cpr_path in cpr_paths]


        vessel_mask_paths = glob.glob(os.path.join(mapping_gen_path, patient+'*'+series.split('_')[0], 'cpr'))[0]
        vessel_mask_paths = sorted(glob.glob(os.path.join(vessel_mask_paths, '*.bmp')))
        vessel_mask_names = [vessel_mask_path.split('/')[-1].split('.')[0] for vessel_mask_path in vessel_mask_paths]
        vessel_mask_files = [numpy.array(Image.open(vessel_mask_path)) for vessel_mask_path in vessel_mask_paths]       

        mapping_paths = glob.glob(os.path.join(mapping_gen_path, patient+'*'+series.split('_')[0], 'cpr'))[0]
        mapping_paths = sorted(glob.glob(os.path.join(mapping_paths, '*.dat')))
        mapping_names =  [mapping_path.split('/')[-1].split('.')[0] for mapping_path in mapping_paths]
        mapping_files = [load_cpr_coord_map(mapping_path) for mapping_path in mapping_paths]


        cta_paths = glob.glob(os.path.join(cta_data_gen_path, patient + '*'+ series.split('_')[0]))[0]
        cta_paths = glob.glob(os.path.join(cta_paths, '*.dcm'))
        cta_paths.sort(key=lambda ele: int(ele.split('/')[-1].split('.')[0]))
        
        #print(glob.glob(os.path.join(cta_paths, '*.dcm')))
        cta_shape = [len(cta_paths), 512, 512]                                               
        
        roi_index_list = []
        for i in range(len(mask_names)):
            if mask_names[i] in mapping_names:        
                roi_index = mapping_names.index(mask_names[i])
                roi_index_list.append(roi_index)
#        print(roi_index_list)
#                print(mask_names[i], mask_files[i].shape, mapping_names[roi_index], mapping_files[roi_index].shape)
        cta_data_reverse = cpr2cta_reverse_map_std_new(mapping_files, mask_files, mask_names, vessel_mask_files, cta_shape, roi_index_list, patient, series)
        draw_save_cta(patient+'_'+series, cta_data_reverse, cta_paths)                
        draw_save_cpr(patient+'_'+series, cpr_paths, mask_files, mask_names, roi_index_list)
#                          
