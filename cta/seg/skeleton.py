import numpy
from skimage.morphology import skeletonize_3d
from lib.utils.union_find import UnionFind


def skeleton(mask):
    ske_mask = skeletonize_3d(mask > 0)
    ske_pts = numpy.asarray(numpy.where(ske_mask > 0)).T
    return ske_pts


def split_skeleton_segments(mask):
    ske_pts = skeleton(mask)
    num_pts = ske_pts.shape[0]
    # print(num_pts)
    w, h, d = mask.shape
    # x, y, z -> index
    indices = numpy.zeros([w, h, d], dtype='int32') - 1
    for i in range(num_pts):
        x, y, z = ske_pts[i]
        indices[x, y, z] = i
    # gen neighbors
    neighbors_list = [[] for i in range(num_pts)]
    for i in range(num_pts):
        x, y, z = ske_pts[i]
        for xj in range(max(0, x - 1), min(x + 2, w)):
            for yj in range(max(0, y - 1), min(y + 2, h)):
                for zj in range(max(0, z - 1), min(z + 2, d)):
                    index_j = indices[xj, yj, zj]
                    if index_j > 0 and index_j != i:
                        neighbors_list[i].append(index_j)
    # fill labels
    labels = numpy.zeros([w, h, d], dtype='int32')
    id_pairs = []
    for i in range(num_pts):
        # not joint points: one group
        if len(neighbors_list[i]) <= 2:
            for nid in neighbors_list[i]:
                if len(neighbors_list[nid]) <= 2:
                    id_pairs.append([i, nid])
        # else:
        #    print(ske_pts[i], [ske_pts[j] for j in neighbors_list[i]])
    # print('id_pairs')
    uf = UnionFind(id_pairs)
    group_ids = uf.run()
    for gid, group in enumerate(group_ids):
        for i in group:
            labels[ske_pts[i, 0], ske_pts[i, 1], ske_pts[i, 2]] = gid + 1
    # print('groups', len(group_ids))
    # fill remaining labels whose all neighbors and joint points
    cur_label = len(group_ids) + 1
    remain_pairs = []
    for i in range(num_pts):
        if labels[ske_pts[i, 0], ske_pts[i, 1], ske_pts[i, 2]] == 0:
            for nid in neighbors_list[i]:
                if labels[ske_pts[nid, 0], ske_pts[nid, 1], ske_pts[nid, 2]] == 0:
                    remain_pairs.append([i, nid])
    # print('remain_pairs', len(remain_pairs))
    uf = UnionFind(remain_pairs)
    group_ids = uf.run()
    for gid, group in enumerate(group_ids):
        neighbors = []
        for i in group:
            neighbors += neighbors_list[i]
        find_index = -1
        for i in set(neighbors):
            if labels[ske_pts[i, 0], ske_pts[i, 1], ske_pts[i, 2]] > 0:
                find_index = labels[ske_pts[i, 0], ske_pts[i, 1], ske_pts[i, 2]]
                break
        if find_index == -1:
            label = cur_label
            cur_label += 1
        else:
            label = find_index
        for i in group:
            labels[ske_pts[i, 0], ske_pts[i, 1], ske_pts[i, 2]] = label
    # print('complete seg pts', len(group_ids), cur_label - 1)
    labeled_pts = [[] for i in range(cur_label - 1)]
    for i in range(num_pts):
        x, y, z = ske_pts[i]
        labeled_pts[labels[x, y, z] - 1].append([x, y, z])
    # judge bridges or ends
    branches_types = ['bridge'] * len(labeled_pts)
    for label in range(len(labeled_pts)):
        for x, y, z in labeled_pts[label]:
            i = indices[x, y, z]
            if len(neighbors_list[i]) == 1:
                branches_types[label] = 'end'
    # print('seg tyeps')
    return ske_pts, labeled_pts, branches_types, labels
