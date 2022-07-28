# file to generate density based clusters which will act as proposals

from fileinput import filename
import numpy as np
import open3d as o3d
import os  
import struct
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

def bin_to_pcd(binFileName):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd

def remove_rear_cloud(pcd):
    pcd_np = np.asarray(pcd.points)
    indices = np.where(pcd_np[:, 0] >= 0)
    infront_cloud = pcd.select_by_index(indices[0])

    ###
    indices = np.where(np.asarray(infront_cloud.points)[:, 0] <= 35)
    infront_cloud = infront_cloud.select_by_index(indices[0])
    ####
    ###
    indices = np.where(np.asarray(infront_cloud.points)[:, 1] <= 25)
    infront_cloud = infront_cloud.select_by_index(indices[0])
    ####
    ###
    indices = np.where(np.asarray(infront_cloud.points)[:, 1] >= -25)
    infront_cloud = infront_cloud.select_by_index(indices[0])
    ####

    return infront_cloud

def segment_ground_plane(pcd, 
                         distance_threshold=0.3,
                         ransac_n=3,
                         num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=ransac_n,
                                         num_iterations=num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([outlier_cloud],
    #                                 zoom=0.8,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])
    return outlier_cloud

def dbscan_cluster(pcd,
                   eps=0.9,
                   min_points=30):
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    max_label = labels.max()
    print(f'point cloud has { max_label + 1 } clusters')
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels == -1] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd],
    #                                 zoom=0.455,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])
    return labels.max() + 1, labels

def get_min_max_box(points):
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    z_min = np.min(points[:, 2])

    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    z_max = np.max(points[:, 2])
    #return np.asarray([[x_min, y]])

def get_bbox(pcd, labels, label_max, filename, folder_name, write=False):
    filename = folder_name + "/bbox_open3d/" +  filename.split('.')[0] + ".txt"
    lis = [pcd]
    for i in range(label_max-1):
        idx = np.where(labels == i)[0]
        curr_points = pcd.select_by_index(idx)
        bbox = curr_points.get_axis_aligned_bounding_box()
        b_points = bbox.get_box_points()
        bbox.color = (1, 0, 0)
        lis.append(bbox)
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        if write:
            with open(filename, "a") as fp:
                fp.write(str(min_bound[0]) + ", ")
                fp.write(str(min_bound[1]) + ", ")
                fp.write(str(min_bound[2]) + ", ")
                fp.write(str(max_bound[0]) + ", ")
                fp.write(str(max_bound[1]) + ", ")
                fp.write(str(max_bound[2]) + "\n")
        #print(bbox, bbox.get_max_bound(), bbox.get_min_bound())
        #break
    #o3d.visualization.draw_geometries(lis)


def load_bin_folder(folder_name):
    count = 0
    total = 0
    for f in tqdm(os.listdir(folder_name)):
        filename = os.path.join(folder_name, f)
        pcd = bin_to_pcd(filename)
        pcd = remove_rear_cloud(pcd)
        pcd_without_ground = segment_ground_plane(pcd)
        label_max, labels = dbscan_cluster(pcd_without_ground)
        get_bbox(pcd_without_ground, labels, label_max, f, "/media/akshay/Data/KITTI/training", True)
        #print(label_max)
        total += 1 
        if label_max <= 50:
            count += 1        

    print("count ", count, "total ", total)

if __name__ == '__main__':
    load_bin_folder("/media/akshay/Data/KITTI/training/velodyne")
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # filename = "Samples_kitti/Samples_kitti/000000.bin"
    # pcd = bin_to_pcd(filename)
    # pcd = remove_rear_cloud(pcd)
    # pcd_without_ground = segment_ground_plane(pcd)
    # label_max, labels = dbscan_cluster(pcd_without_ground)
    #get_bbox(pcd_without_ground, labels, label_max, vis)
    #o3d.visualization.draw_geometries([pcd_without_ground])
