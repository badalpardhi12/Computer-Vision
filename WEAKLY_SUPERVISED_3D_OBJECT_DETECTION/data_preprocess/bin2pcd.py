# To convert bin lidar files to pcd for KITTI dataset

import numpy as np
import struct
import os
import sys
import open3d as o3d


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


if __name__ == "__main__":
    binFolderName = "/media/akshay/Data/KITTI/testing/velodyne/"
    pcdFolderName = "/media/akshay/Data/KITTI/testing/velodyne_pcd/"
    for i in os.listdir(binFolderName):
        binFileName = binFolderName+i
        print(i)
        pcd = bin_to_pcd(binFileName)
        pcd_np = np.asarray(pcd.points)
        valid_x = np.where(pcd_np[:, 0] >= 0)
        pcd_np = pcd_np[valid_x]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcdFileName = pcdFolderName+i[:-4]+'.pcd'
        print(pcdFileName)
        o3d.io.write_point_cloud(pcdFileName, pcd)
    