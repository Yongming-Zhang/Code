from seg_util import *
from lib.utils.mio import *


def main():
    # base_dir = '/breast_data/cta/new_data/vessel_seg/with_sg_407_plaque/'
    # subjects = load_string_list(base_dir + 'config/train_list.txt')
    # subjects += load_string_list(base_dir + 'config/valid_test_list.txt')
    base_dir = '/breast_data/cta/new_data/vessel_seg/all_3580/'
    subjects = load_string_list(base_dir + 'config/train_list.txt')
    subjects += load_string_list(base_dir + 'config/valid_list.txt')
    subjects += load_string_list(base_dir + 'config/test_list.txt')
    base_save_dir = base_dir + 'norm_z0.5/'
    image_dir, mask_dir = base_save_dir + 'image/', base_save_dir + 'mask/'
    heatmap_dir, ske_dir = base_save_dir + 'heatmap/', base_save_dir + 'skeleton/'
    mkdir_safe(image_dir)
    mkdir_safe(mask_dir)
    mkdir_safe(heatmap_dir)
    mkdir_safe(ske_dir)
    for i, subject in enumerate(sorted(subjects)):
        # vsdl = VesselSegDataLoader(subject, mask_dir=base_dir + 'mask_wp4/')
        vsdl = VesselSegDataLoader(subject)
        image_nii, mask_nii, heatmap_nii, points_dict = vsdl.save_data(
            image_dir + '%s.nii' % subject,
            mask_dir + '%s.nii' % subject,
            heatmap_dir + '%s.nii' % subject,
            ske_dir + '%s.mat' % subject
        )
        print('No.%d/%d, %s, %s, %s' % (i, len(subjects), subject, str(image_nii.header['pixdim'][:4]), str(image_nii.shape)))


if __name__ == '__main__':
    main()

