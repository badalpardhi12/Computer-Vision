from itertools import count
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os  
from dataprocess import load_bbox, filter_boxes
import ctypes
from tqdm import tqdm
from kitti_utils import get_pixel_coordinates, get_pixel_coordinates_and_3d
from PIL import Image
from torchvision import transforms

class KITTIBEV(Dataset):
    def __init__(self, 
                is_train=True,
                lidar_folder_name=None,
                label_folder_name=None,
                valid_data_list_filename=None):

        self.class_names = ['Car', 
                            'Others']

        self.geometry = {'L1': -20.0,
                        'L2': 20.0,
                        'W1': 0.0,
                        'W2': 35.0,
                        'H1': -2.5,
                        'H2': 1.0,
                        'input_shape': (400, 350, 36),
                        'label_shape': (200, 175, 7)}
        self.req_img_size = (400, 350)
        self.use_npy = False
        self.num_classes = len(self.class_names)
        self.lidar_folder_name = lidar_folder_name
        self.label_folder_name = label_folder_name
        self.LidarLib = ctypes.cdll.LoadLibrary('./preprocess/LidarPreprocess.so')
        self.is_train = is_train
        if self.is_train:
            self.sub_folder = 'training'
        else:
            self.sub_folder = 'testing'
        filenames_list = []

        with open(valid_data_list_filename, "r") as f: 
            for line in f.readlines():
                line = line.split("\n")[0]
                filenames_list.append(line)
        
        self.inv_class = {i: class_name for i, class_name in enumerate(self.class_names)}
        self.class_to_int = {class_name: i for i, class_name in enumerate(self.class_names)}

        ####
        self.preload_proposals = []
        self.preload_labels = []
        self.preload_gt_boxes = []
        self.preload_gt_class_list = []
        self.filenames_list = []

        count = 0
        mi = -100000
        ccx = 0
        prop_max = -1
        prop_min = 10000
        print("Preloading Data")
        for filename in tqdm(filenames_list,
                            total=len(filenames_list),
                            leave=False):
            labels, gt_boxes, gt_class_list = self.get_labels(filename)
            if gt_boxes.shape[0] == 0:
                filenames_list.remove(filename)
                ccx += 1
                continue
            self.preload_labels.append(labels)
            self.preload_gt_boxes.append(gt_boxes)
            self.preload_gt_class_list.append(gt_class_list)
            self.filenames_list.append(filename)
            proposals = self.get_proposals(filename)
            assert(gt_class_list.shape[0] == gt_class_list.shape[0])
            assert np.min(proposals[:, 0]) >= 0, "x_min"
            assert np.max(proposals[:, 2]) < 400, "x_max"
            assert np.max(proposals[:, 3]) < 350, "y_max"
            assert np.min(proposals[:, 1]) >= 0
            self.preload_proposals.append(proposals)
            prop_max = max(prop_max, proposals.shape[0])
            prop_min = min(prop_min, proposals.shape[0])
        ####
        print(prop_max, prop_min)

    def __len__(self) -> int:
        return len(self.filenames_list)

    def __getitem__(self, index: int):
        filename = self.filenames_list[index]
        bev = self.load_velo_scan(index)
        #bev = self.lidar_preprocess(bev)
        bev = bev.transpose(2, 0, 1)
        proposals = self.preload_proposals[index]
        labels = self.preload_labels[index] 
        gt_boxes = self.preload_gt_boxes[index]
        gt_class_list = self.preload_gt_class_list[index] # .get_labels(self.filenames_list[index])
        return {'bev': torch.from_numpy(bev),
                'labels': torch.from_numpy(labels),
                'gt_boxes': torch.from_numpy(gt_boxes),
                'proposals': torch.from_numpy(proposals),
                'gt_class_list': torch.from_numpy(gt_class_list)}

    def augment_pcl(self, pcl_boxes):
        np.random.seed(200)
        pcl_boxes_aranged = np.concatenate([pcl_boxes[:, 1].reshape(-1, 1),
                                            pcl_boxes[:, 0].reshape(-1, 1), 
                                            pcl_boxes[:, 4].reshape(-1, 1), 
                                            pcl_boxes[:, 3].reshape(-1, 1)], axis=1)

        pcl_boxes = self.scale_bev(pcl_boxes_aranged)
        valid_idx_in_ = np.where(pcl_boxes[:, 0] < 400)
        pcl_boxes = pcl_boxes[valid_idx_in_]
        valid_idx_in_ = np.where(pcl_boxes[:, 2] >= 0)
        pcl_boxes = pcl_boxes[valid_idx_in_]
        valid_idx_in_ = np.where(pcl_boxes[:, 1] <= 349)
        pcl_boxes = pcl_boxes[valid_idx_in_]

        rand_values = np.random.uniform(low=1., high=2.5, size=(3))
        total_boxes = np.zeros((0, 4))
        for rand_value in [1, 1.2, 1.5, 3.2]:
            #####
            augment_distance = (pcl_boxes[:, 3] - pcl_boxes[:, 1]).reshape(-1, 1) * rand_value # N*1
            new_y_max = pcl_boxes[:, 1].reshape(-1, 1) + augment_distance
            new_y_max = np.where(new_y_max[:, 0] >= 350, 349, new_y_max[:, 0])
            #####

            #####
            if rand_value == 3.2:
                augment_distance = (pcl_boxes[:, 2] - pcl_boxes[:, 0]).reshape(-1, 1) * rand_value/4
            else:
                 augment_distance = (pcl_boxes[:, 2] - pcl_boxes[:, 0]).reshape(-1, 1) * rand_value/2 # N*1
            new_x_max = pcl_boxes[:, 0].reshape(-1, 1) + augment_distance
            new_x_max = np.where(new_x_max[:, 0] >= 400, 399, new_x_max[:, 0])
            new_x_min = pcl_boxes[:, 0].reshape(-1, 1) - augment_distance
            new_x_min = np.where(new_x_min[:, 0] <= 0, 0, new_x_min[:, 0])
            #####
            
            augmented_box = np.concatenate([new_x_min.reshape(-1, 1),
                                            pcl_boxes[:, 1].reshape(-1, 1),
                                            new_x_max.reshape(-1, 1),
                                            new_y_max.reshape(-1, 1)], axis = 1)
            total_boxes = np.concatenate([total_boxes, augmented_box], axis=0)
            # combined_augmented_boxes = np.concatenate([combined_boxes[:, 1].reshape(-1, 1),
            #                                             combined_boxes[:, 0].reshape(-1, 1), 
            #                                         combined_boxes[:, 4].reshape(-1, 1),
            #                                         new_y_max.reshape(-1, 1)], axis=1)
            # combined_augmented_boxes_2d = np.concatenate([combined_boxes_2d,
            #                                             combined_augmented_boxes], axis=0)
            # combined_augmented_boxes_2d = self.scale_bev(combined_augmented_boxes_2d)
        return total_boxes

    def get_proposals(self, filename):
        pcl_proposal_filename = os.path.join(self.lidar_folder_name,
                                             self.sub_folder,
                                             "bbox_pcl",
                                             filename + ".txt")
        dbscan_proposal_filename = os.path.join(self.lidar_folder_name,
                                                self.sub_folder,
                                                "bbox_open3d",
                                                filename + ".txt")
        pcl_boxes = load_bbox(pcl_proposal_filename, pcl=True)
        pcl_boxes[:, 1] = np.where(pcl_boxes[:, 1] <= -40, -40, pcl_boxes[:, 1]) #np.max(pcl_boxes[:, 2], -40)
        pcl_boxes[:, 3] = np.where(pcl_boxes[:, 3] >= 70, 69.9, pcl_boxes[:, 3]) #np.min(pcl_boxes[:, 3], 70)
        pcl_boxes[:, 4] = np.where(pcl_boxes[:, 4] >= 40, 39.9, pcl_boxes[:, 4]) #np.min(pcl_boxes[:, 4], 40)

        pcl_augmented = self.augment_pcl(pcl_boxes)
        dbscan_boxes = load_bbox(dbscan_proposal_filename, pcl=False) 
        # dbscan_filtered_boxes = filter_boxes(dbscan_boxes,
        #                                     100 - pcl_boxes.shape[0])
        combined_boxes = np.concatenate([pcl_boxes, dbscan_boxes], axis=0)

        # check row order
        combined_boxes_2d = np.concatenate([combined_boxes[:, 1].reshape(-1, 1),
                                            combined_boxes[:, 0].reshape(-1, 1), 
                                            combined_boxes[:, 4].reshape(-1, 1), 
                                            combined_boxes[:, 3].reshape(-1, 1)], axis=1)

        ##### box augmentation
        augment_distance = (combined_boxes[:,3] - combined_boxes[:, 0]).reshape(-1, 1) * 2 # N*1
        new_y_max = combined_boxes[:, 0].reshape(-1, 1) + augment_distance
        new_y_max = np.where(new_y_max[:, 0] >= 70, 69.9, new_y_max[:, 0])
        #####
        combined_augmented_boxes = np.concatenate([combined_boxes[:, 1].reshape(-1, 1),
                                                   combined_boxes[:, 0].reshape(-1, 1), 
                                                   combined_boxes[:, 4].reshape(-1, 1),
                                                   new_y_max.reshape(-1, 1)], axis=1)
        combined_augmented_boxes_2d = np.concatenate([combined_boxes_2d,
                                                     combined_augmented_boxes], axis=0)
        combined_augmented_boxes_2d = self.scale_bev(combined_augmented_boxes_2d)

        ######## cropping
        valid_idx_in_ = np.where(combined_augmented_boxes_2d[:, 0] < 400)
        combined_augmented_boxes_2d = combined_augmented_boxes_2d[valid_idx_in_]
        valid_idx_in_ = np.where(combined_augmented_boxes_2d[:, 2] >= 0)
        combined_augmented_boxes_2d = combined_augmented_boxes_2d[valid_idx_in_]
        valid_idx_in_ = np.where(combined_augmented_boxes_2d[:, 1] <= 349)
        combined_augmented_boxes_2d = combined_augmented_boxes_2d[valid_idx_in_]
        combined_augmented_boxes_2d = np.concatenate([combined_augmented_boxes_2d, pcl_augmented], axis=0)
        combined_augmented_boxes_2d[:, 0] = np.where(combined_augmented_boxes_2d[:, 0] < 0, 
                                            0, combined_augmented_boxes_2d[:, 0])
        combined_augmented_boxes_2d[:, 0] = np.where(combined_augmented_boxes_2d[:, 1] < 0, 
                                            0, combined_augmented_boxes_2d[:, 1])
        combined_augmented_boxes_2d[:, 2] = np.where(combined_augmented_boxes_2d[:, 2] >= 400,
                                            399, combined_augmented_boxes_2d[:, 2])
        combined_augmented_boxes_2d[:, 3] = np.where(combined_augmented_boxes_2d[:, 3] >= 350, 
                                            349, combined_augmented_boxes_2d[:, 3])

        return combined_augmented_boxes_2d
    
    def lidar_preprocess(self, scan):
        
        velo = scan
        velo_processed = np.zeros(self.geometry['input_shape'], dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        for i in range(velo.shape[0]):
            if self.point_in_roi(velo[i, :]):
                x = int((velo[i, 1]-self.geometry['L1']) / 0.1)
                y = int((velo[i, 0]-self.geometry['W1']) / 0.1)
                z = int((velo[i, 2]-self.geometry['H1']) / 0.1)
                velo_processed[x, y, z] = 1
                velo_processed[x, y, -1] += velo[i, 3]
                intensity_map_count[x, y] += 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count, \
                        where=intensity_map_count!=0)
        return velo_processed
    
    def point_in_roi(self, point):
        if (point[0] - self.geometry['W1']) < 0.01 or (self.geometry['W2'] - point[0]) < 0.01:
            return False
        if (point[1] - self.geometry['L1']) < 0.01 or (self.geometry['L2'] - point[1]) < 0.01:
            return False
        if (point[2] - self.geometry['H1']) < 0.01 or (self.geometry['H2'] - point[2]) < 0.01:
            return False
        return True

    def get_labels(self, 
                  filename,
                  map_height=400):
        label = np.zeros((self.num_classes,))
        gt_boxes = np.zeros((0, 4))
        gt_class_list = []
        label_filename = os.path.join(self.lidar_folder_name,
                                      self.sub_folder,
                                      "label_2", 
                                      filename + ".txt")
        with open(label_filename, "r") as f:
            for line in f.readlines():
                x = line.split(" ")
                if x[0] == 'DontCare':
                    continue
                
                curr_box_labels = [float(x[i]) for i in range(8, 15)]
                gt_box_curr = self.get_gt_bbox(curr_box_labels)

                #####
                ###change this, in bev return and proposal return and wandb
                if gt_box_curr[0, 1] >= 350 or \
                   gt_box_curr[0, 2] < 0 or \
                   gt_box_curr[0, 0] >= 400:
                    continue
                gt_box_curr[0, 3] = np.where(gt_box_curr[0, 3] >= 350, 349, gt_box_curr[0, 3])
                gt_box_curr[0, 0] = np.where(gt_box_curr[0, 0] <= 0, 0, gt_box_curr[0, 0])
                gt_box_curr[0, 2] = np.where(gt_box_curr[0, 2] >= 400, 399, gt_box_curr[0, 2]) 
                if x[0] == 'Car':
                    label[0] = 1
                    gt_class_list.append(0)
                else:
                    label[1] = 1
                    gt_class_list.append(1)
                
                ######
                # if gt_box_curr[0, 0] < 200 or gt_box_curr[0, 2] >= 600:
                #     print(gt_box_curr[0, 0], gt_box_curr[0, 2], gt_box_curr[0, 3])

                gt_boxes = np.concatenate([gt_boxes, gt_box_curr], axis=0)
        return label, gt_boxes, np.asarray(gt_class_list).reshape(-1)

    def scale_bev(self, bev, map_height=400):
        bev_new = bev / 0.1
        bev_new[:, 0] += int(map_height // 2)
        bev_new[:, 2] += int(map_height // 2)
        # bev_new[:, 0] = map_height - bev_new[:, 0]
        # bev_new[:, 2] = map_height - bev_new[:, 2]
        return bev_new

    def get_gt_bbox(self, bbox):
        w, h, l, y, z, x, yaw = bbox
        y = -y
        yaw = -(yaw + np.pi / 2)
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        max_x = np.max(bev_corners[:, 0])
        min_x = np.min(bev_corners[:, 0])
        max_y = np.max(bev_corners[:, 1])
        min_y = np.min(bev_corners[:, 1])

        bev = np.array([min_y, min_x, max_y, max_x]).reshape(-1, 4)
        bev = self.scale_bev(bev)
        
        ####
        # bev_inverted = torch.tensor([bev[0, 1], bev[0, 0], bev[0, 3], bev[0, 2]]).reshape(1, -1)
        # return bev_inverted
        ####
        return bev

    def load_velo_scan(self, index):
        """Helper method to parse velodyne binary files into a list of scans."""
        filename = os.path.join(self.lidar_folder_name,
                                self.sub_folder,
                                "velodyne", 
                                self.filenames_list[index] + ".bin")

        if self.use_npy:
            scan = np.load(filename[:-4]+'.npy')
        else:
            c_name = bytes(filename, 'utf-8')
            scan = np.zeros(self.geometry['input_shape'], dtype=np.float32)
            c_data = ctypes.c_void_p(scan.ctypes.data)
            self.LidarLib.createTopViewMaps(c_data, c_name)
            #scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        return scan

class KITTICam(Dataset):
    def __init__(self, 
                is_train=True,
                lidar_folder_name=None,
                label_folder_name=None,
                valid_data_list_filename=None,
                req_img_size=(1242, 375)):
        self.x_min = -25
        self.x_max = 25
        self.y_min = 0
        self.y_max = 35
        self.req_img_size = req_img_size
        self.initial_image_size = (1242, 375)

        self.class_names = ['Car']
        self.num_classes = len(self.class_names)
        self.lidar_folder_name = lidar_folder_name
        self.label_folder_name = label_folder_name
        self.is_train = is_train
        if self.is_train:
            self.sub_folder = 'training'
        filenames_list = []

        with open(valid_data_list_filename, "r") as f: 
            for line in f.readlines():
                line = line.split("\n")[0]
                filenames_list.append(line)
        
        self.inv_class = {i: class_name for i, class_name in enumerate(self.class_names)}
        self.class_to_int = {class_name: i for i, class_name in enumerate(self.class_names)}

        ####
        self.preload_proposals = []
        self.preload_labels = []
        self.preload_gt_boxes = []
        self.preload_gt_class_list = []
        self.filenames_list = []
        self.preload_proposals_3d = []
        self.preload_gt_boxes_3d = []
        self.transforms = transforms.Compose([transforms.Resize((self.req_img_size[1], self.req_img_size[0])),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        print("Preloading Data")
        for filename in tqdm(filenames_list,
                            total=len(filenames_list),
                            leave=False):
            #proposals, gt_boxes = get_pixel_coordinates(filename, lidar_folder_name)
            proposals, gt_boxes, proposals_3d, gt_boxes_3d = get_pixel_coordinates_and_3d(filename,
                                                                    lidar_folder_name,
                                                                    isplot=True)
            if proposals.shape[0] == 0:
                continue
            if proposals.shape[0] != 0:
                proposals[:, 0] = proposals[:, 0] * (self.req_img_size[0] / self.initial_image_size[0])
                proposals[:, 1] = proposals[:, 1] * (self.req_img_size[1] / self.initial_image_size[1])
                proposals[:, 2] = proposals[:, 2] * (self.req_img_size[0] / self.initial_image_size[0])
                proposals[:, 3] = proposals[:, 3] * (self.req_img_size[1] / self.initial_image_size[1])

            if gt_boxes.shape[0] != 0:
                gt_boxes[:, 0] = gt_boxes[:, 0] * (self.req_img_size[0] / self.initial_image_size[0])
                gt_boxes[:, 1] = gt_boxes[:, 1] * (self.req_img_size[1] / self.initial_image_size[1])
                gt_boxes[:, 2] = gt_boxes[:, 2] * (self.req_img_size[0] / self.initial_image_size[0])
                gt_boxes[:, 3] = gt_boxes[:, 3] * (self.req_img_size[1] / self.initial_image_size[1])


            label = np.zeros((self.num_classes, ))
            label[0] = 1 if gt_boxes.shape[0] != 0 else 0
            gt_class_list = np.zeros((gt_boxes.shape[0], ))
            self.preload_labels.append(label)
            self.preload_gt_class_list.append(gt_class_list)
            self.filenames_list.append(filename)

            # self.preload_gt_boxes.append(gt_boxes)
            # self.preload_proposals.append(proposals)
            
            ###
            self.preload_gt_boxes.append(gt_boxes)
            self.preload_proposals.append(proposals)
            self.preload_gt_boxes_3d.append(gt_boxes_3d)
            self.preload_proposals_3d.append(proposals_3d)
            ###

        ####
    def __len__(self) -> int:
        return len(self.filenames_list)

    def __getitem__(self, index: int):
        #print(index)
        image_path = os.path.join(self.lidar_folder_name, 
                                "KITTI",
                                self.sub_folder,
                                "image_2",
                                self.filenames_list[index] + ".png") 
        image = Image.open(image_path)
        image = self.transforms(image)
        labels = self.preload_labels[index]
        gt_boxes = self.preload_gt_boxes[index]
        proposals = self.preload_proposals[index]
        gt_class_list = self.preload_gt_class_list[index]
        gt_boxes_3d = self.preload_gt_boxes_3d[index]
        proposals_3d = self.preload_proposals_3d[index]
        
        ####scan
        filename = os.path.join(self.lidar_folder_name,
                                "KITTI",
                                self.sub_folder,
                                "velodyne", 
                                self.filenames_list[index] + ".bin")

        scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        ####
        return {'filename': self.filenames_list[index],
                'image': image,
                'labels': torch.from_numpy(labels),
                'gt_boxes': torch.from_numpy(gt_boxes),
                'proposals': torch.from_numpy(proposals),
                'gt_class_list': torch.from_numpy(gt_class_list),
                'gt_boxes_3d': torch.from_numpy(gt_boxes_3d),
                'proposals_3d': torch.from_numpy(proposals_3d),
                'scan': scan}

    def get_label(self, filename):
        label = np.zeros((self.num_classes,))
        gt_class_list = []
        label_filename = os.path.join(self.lidar_folder_name,
                                      self.sub_folder,
                                      "label_2", 
                                      filename + ".txt")
        with open(label_filename, "r") as f:
            for line in f.readlines():
                x = line.split(" ")
                if x[0] == 'Car':
                    label[0] = 1
                    gt_class_list.append(0)
        return label, gt_class_list

if __name__ == "__main__":
    valid_data_list_filename = "./valid_data_list_after_threshold.txt"
    lidar_folder_name = "./data"
    dataset = KITTIBEV(valid_data_list_filename=valid_data_list_filename, 
                       lidar_folder_name=lidar_folder_name)
    dataset[10]