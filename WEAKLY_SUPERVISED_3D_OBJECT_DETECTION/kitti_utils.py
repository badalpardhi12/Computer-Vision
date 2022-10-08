import numpy as np
import cv2
import os, math
from scipy.optimize import leastsq
from PIL import Image

def read_calib_file(filepath):
    """ Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def compute_box_3d(dimensions, location, rot_y):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(rot_y)

    # 3d bounding box dimensions l, w, h starting
    h = dimensions[0]
    w = dimensions[1]
    l = dimensions[2]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] +  location[0]
    corners_3d[1, :] = corners_3d[1, :] +  location[1]
    corners_3d[2, :] = corners_3d[2, :] +  location[2]
    # print 'cornsers_3d: ', corners_3d
    return np.transpose(corners_3d)

def project_velo_to_ref(pts_3d_velo, V2C):
    pts_3d_velo = cart2hom(pts_3d_velo)  # nx4
    return np.dot(pts_3d_velo, np.transpose(V2C))

def project_ref_to_rect(pts_3d_ref, R0):
    """ Input and Output are nx3 points """
    return np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))

def project_velo_to_rect(pts_3d_velo, V2C, R0):
    pts_3d_ref = project_velo_to_ref(pts_3d_velo, V2C)
    return project_ref_to_rect(pts_3d_ref, R0)

def project_rect_to_image(pts_3d_rect, P):
    """ Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    """
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = np.dot(pts_3d_rect, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Output: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def project_velo_to_image(pts_3d_velo, P, V2C, R0):
    """ Input: nx3 points in velodyne coord.
        Output: nx2 points in image2 coord.
    """
    pts_3d_rect = project_velo_to_rect(pts_3d_velo, V2C, R0)
    return project_rect_to_image(pts_3d_rect, P)

def get_8points_from_6points(point_list):
    x_min = point_list[0]
    y_min = point_list[1]
    z_min = point_list[2]
    x_max = point_list[3]
    y_max = point_list[4]
    z_max = point_list[5]

    eight_points = np.empty((8, 3))
    eight_points[0] = [x_min, y_min, z_min]
    eight_points[1] = [x_min, y_min, z_max]
    eight_points[2] = [x_min, y_max, z_min]
    eight_points[3] = [x_min, y_max, z_max]
    eight_points[4] = [x_max, y_min, z_min]
    eight_points[5] = [x_max, y_min, z_max]
    eight_points[6] = [x_max, y_max, z_min]
    eight_points[7] = [x_max, y_max, z_max]

    return eight_points

def eight_points_to_4points_image(eight_points):
    four_points = np.empty(4)
    four_points[0] = np.min(eight_points[:, 0])
    four_points[1] = np.min(eight_points[:, 1])
    four_points[2] = np.max(eight_points[:, 0])
    four_points[3] = np.max(eight_points[:, 1])

    return four_points

def read_proposals(proposals_file, P, V2C, R0, isplot=False):
    """ Reads a KITTI detection proposal file.
        Output:
            dets: list of detections
    """
    with open(proposals_file, 'r') as f:
        lines = f.readlines()
        dets = [line.split(',') for line in lines]
        boxes_6_points = []
        #####
        for line in lines:
            curr_box_6points = np.array([float(a) for a in line.split(',')])
            #curr_box_6points = refine_proposals(curr_box_6points.reshape(1, -1)).reshape(-1)
            if curr_box_6points.shape[0] != 0:
                boxes_6_points.append(curr_box_6points)
        #####
        dets = [get_8points_from_6points(box) for box in boxes_6_points]

    image_3d = [project_velo_to_image(det, P, V2C, R0) for det in dets]
    img_coords = [eight_points_to_4points_image(det) for det in image_3d]
    if isplot:
        return img_coords, boxes_6_points
    return img_coords

def read_label(label_file, req_label = 'Car'):
    """ Reads a KITTI detection label file.
        Output:
            list of dict, with keys: 'type', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y'
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
        list_gt = [line.split(' ') for line in lines]
        gts = []
        for gt in list_gt:
            gt_dict = {}
            gt_dict['type'] = gt[0]
            if gt[0] != 'Car' and gt[0] != 'Van':
                continue
            gt_dict['truncated'] = float(gt[1])
            gt_dict['occluded'] = float(gt[2])
            gt_dict['alpha'] = float(gt[3])
            gt_dict['bbox'] = [float(gt[4]), float(gt[5]), float(gt[6]), float(gt[7])]
            gt_dict['dimensions'] = [float(gt[8]), float(gt[9]), float(gt[10])]
            gt_dict['location'] = [float(gt[11]), float(gt[12]), float(gt[13])]
            gt_dict['rotation_y'] = float(gt[14])
            gts.append(gt_dict)
        
    return gts

def get_2D_gt_boxes(label_file):
    gts = read_label(label_file)
    boxes = []
    for gt in gts:
        box = gt['bbox']
        boxes.append(box)
    return boxes
        

def read_image(image_file):
    """ Reads and returns an image by file path. """
    return cv2.imread(image_file)

def plot_2d_boxes(img, boxes, color=(0, 255, 0)):
    """ Plot 2D bounding boxes on image. """
    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    return img

def test():
    for i in range(100):
        filename = '%06d' % i
        calib = read_calib_file("KITTI/training/calib/{}.txt".format(filename))

        P = calib['P2']
        P = np.reshape(P, [3,4])
        V2C = calib['Tr_velo_to_cam']
        V2C = np.reshape(V2C, [3,4])
        R0 = calib["R0_rect"]
        R0 = np.reshape(R0, [3,3])

        gt_boxes = get_2D_gt_boxes("KITTI/training/label_2/{}.txt".format(filename))
        boxes1 = read_proposals("KITTI/training/bbox_pcl/{}.txt".format(filename), P, V2C, R0)
        boxes2 = read_proposals("KITTI/training/bbox/{}.txt".format(filename), P, V2C, R0)
        image = read_image("KITTI/training/image_2/{}.png".format(filename))
        
        img = plot_2d_boxes(image, boxes1, color=(0,0,255))
        img = plot_2d_boxes(img, boxes2, color=(255,0,0))
        img = plot_2d_boxes(img, gt_boxes, color=(0,255,0))


        cv2.imshow("image_{}".format(i), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_pixel_coordinates(filename, basefilename="./data/"):
    
    '''
        The filename needs to be a string wihout any extension (ex. 000000)
        The function returns two numpy arrays with the pixel coordinates of proposals and gt_boxes repectively
        Put the paths carefully for each file
    '''
    calib = read_calib_file(basefilename + "KITTI/training/calib/{}.txt".format(filename))

    P = calib['P2']
    P = np.reshape(P, [3,4])
    V2C = calib['Tr_velo_to_cam']
    V2C = np.reshape(V2C, [3,4])
    R0 = calib["R0_rect"]
    R0 = np.reshape(R0, [3,3])

    gt_boxes = np.array(get_2D_gt_boxes(basefilename + "KITTI/training/label_2/{}.txt".format(filename)))
    gt_velo_boxes = get_velo_from_cam(filename, basefilename).reshape(-1, 6)
    gt_boxes = refine_gt_boxes(gt_boxes, gt_velo_boxes)

    boxes1 = np.array(read_proposals(basefilename + "KITTI/training/bbox_pcl/{}.txt".format(filename), P, V2C, R0))
    #boxes1 = refine_proposals(boxes1)
    boxes2 = np.array(read_proposals(basefilename + "KITTI/training/bbox_open3d/{}.txt".format(filename), P, V2C, R0))

    proposals = np.concatenate((boxes1.reshape(-1, 4), boxes2.reshape(-1, 4)), axis=0) # x_min, y_min, x_max, y_max (n x 4)
    proposals = refine_image_proposals(proposals)
    return proposals, gt_boxes


def get_pixel_coordinates_and_3d(filename, basefilename="./data/", isplot=False):
    
    '''
        The filename needs to be a string wihout any extension (ex. 000000)
        The function returns two numpy arrays with the pixel coordinates of proposals and gt_boxes repectively
        Put the paths carefully for each file
    '''
    calib = read_calib_file(basefilename + "KITTI/training/calib/{}.txt".format(filename))

    P = calib['P2']
    P = np.reshape(P, [3,4])
    V2C = calib['Tr_velo_to_cam']
    V2C = np.reshape(V2C, [3,4])
    R0 = calib["R0_rect"]
    R0 = np.reshape(R0, [3,3])

    gt_boxes = np.array(get_2D_gt_boxes(basefilename + "KITTI/training/label_2/{}.txt".format(filename)))
    gt_velo_boxes = get_velo_from_cam(filename, basefilename).reshape(-1, 6)
    gt_boxes, gt_boxes_3d = refine_gt_boxes(gt_boxes, gt_velo_boxes, isplot)

    boxes1, boxes1_3d = read_proposals(basefilename + "KITTI/training/bbox_pcl/{}.txt".format(filename), 
                                       P, V2C, R0, isplot)
    boxes1 = np.asarray(boxes1)
    boxes1_3d = np.asarray(boxes1_3d)
    #boxes1 = refine_proposals(boxes1)
    boxes2, boxes2_3d = read_proposals(basefilename + "KITTI/training/bbox_open3d/{}.txt".format(filename),
                                       P, V2C, R0, isplot)
    boxes2 = np.asarray(boxes2)
    boxes2_3d = np.asarray(boxes2_3d)

    proposals = np.concatenate((boxes1.reshape(-1, 4), boxes2.reshape(-1, 4)), axis=0) # x_min, y_min, x_max, y_max (n x 4)
    proposals_3d = np.concatenate((boxes1_3d.reshape(-1, 6),
                                   boxes2_3d.reshape(-1, 6)), axis=0)
    proposals, proposals_3d = refine_image_proposals(proposals, proposals_3d, isplot)
    if isplot:
        ###
        gt_boxes_3d_n = gt_boxes_3d.copy()
        gt_boxes_3d_n[:, 1] = gt_boxes_3d[:, 2]
        gt_boxes_3d_n[:, 2] = gt_boxes_3d[:, 1]
        gt_boxes_3d_n[:, 4] = gt_boxes_3d[:, 5]
        gt_boxes_3d_n[:, 5] = gt_boxes_3d[:, 4]

        ###
        return proposals, gt_boxes, proposals_3d, gt_boxes_3d
    return proposals, gt_boxes


def refine_gt_boxes(gt_boxes, gt_velo_boxes, isplot=False):
    x_min = 0
    x_max = 35
    y_min = -25
    y_max = 25
    
    ##### Limiting along axi
    refined_idx = np.where(gt_velo_boxes[:, 0] < x_max)
    gt_velo_boxes = gt_velo_boxes[refined_idx]
    gt_boxes = gt_boxes[refined_idx]
    refined_idx = np.where(gt_velo_boxes[:, 1] < y_max)
    gt_velo_boxes = gt_velo_boxes[refined_idx]
    gt_boxes = gt_boxes[refined_idx]

    refined_idx = np.where(gt_velo_boxes[:, 4] > y_min)
    gt_velo_boxes = gt_velo_boxes[refined_idx]
    gt_boxes = gt_boxes[refined_idx]
    #####
    if gt_boxes.shape[0] == 0:
        if isplot:
            return np.zeros((0, 4)), np.zeros((0, 6))
        else:
            return np.zeros((0, 4))

    if isplot:
        return gt_boxes, gt_velo_boxes
    return gt_boxes

def refine_image_proposals(proposals, proposals_3d, isplot=False):
    x_min = 0
    x_max = 1224
    y_min = 0
    y_max = 375

    ##### Limiting along axis
    refined_idx = np.where(proposals[:, 0] < x_max)
    refined_proposals = proposals[refined_idx].reshape(-1, 4)
    proposals_3d = proposals_3d[refined_idx].reshape(-1, 6)

    refined_idx = np.where(refined_proposals[:, 1] < y_max)
    refined_proposals = refined_proposals[refined_idx].reshape(-1, 4)
    proposals_3d = proposals_3d[refined_idx].reshape(-1, 6)

    refined_idx = np.where(refined_proposals[:, 2] > x_min)
    refined_proposals = refined_proposals[refined_idx].reshape(-1, 4)
    proposals_3d = proposals_3d[refined_idx].reshape(-1, 6)

    refined_idx = np.where(refined_proposals[:, 3] > y_min)
    refined_proposals = refined_proposals[refined_idx].reshape(-1, 4)
    proposals_3d = proposals_3d[refined_idx].reshape(-1, 6)
    ####

    ##### compressing the boxes4
    refined_proposals[:, 0] = np.where(refined_proposals[:, 0] < x_min,
                                x_min, refined_proposals[:, 0])
    ####
    refined_proposals[:, 1] = np.where(refined_proposals[:, 1] < y_min, 
                                y_min, refined_proposals[:, 1])
    ####
    refined_proposals[:, 2] = np.where(refined_proposals[:, 2] > x_max,
                                x_max-1, refined_proposals[:, 2])
    ###
    refined_proposals[:, 3] = np.where(refined_proposals[:, 3] > y_max,
                                y_max-1, refined_proposals[:, 3])
    #####
    if isplot:
        return refined_proposals, proposals_3d
    return refined_proposals

def get_velo_from_cam(filename, basefilename="./data/"):
    '''
        The filename needs to be a string wihout any extension (ex. 000000)
        The function return mxnx3 numpy array with the velodyne coordinates of gt_boxes
        Here m is number of gt_boxes and n is 8
        Put the paths carefully for each file
    '''
    calib = read_calib_file(basefilename + "KITTI/training/calib/{}.txt".format(filename))

    P = calib['P2']
    P = np.reshape(P, [3,4])
    V2C = calib['Tr_velo_to_cam']
    V2C = np.reshape(V2C, [3,4])
    C2V = inverse_rigid_trans(V2C)
    R0 = calib["R0_rect"]
    R0 = np.reshape(R0, [3,3])

    label = read_label(basefilename + "KITTI/training/label_2/{}.txt".format(filename))

    velo_gts = []
    for gt in label:
        cam_box_3d = compute_box_3d(gt["dimensions"], gt["location"], gt["rotation_y"])
        velo_box_3d = cam_to_velo(cam_box_3d, R0, C2V)
        velo_box_limit = get_6points_from_8points(velo_box_3d)
        velo_gts.append(velo_box_limit)

    return np.array(velo_gts)

def cam_to_velo(pts_3d_rect, R0, C2V):
    pts_3d_ref = project_rect_to_ref(pts_3d_rect, R0)
    pts_3d_velo = project_ref_to_velo(pts_3d_ref, C2V)
    return pts_3d_velo

def project_rect_to_ref(pts_3d_rect, R0):
    """ Input and Output are nx3 points """
    return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))

def project_ref_to_velo(pts_3d_ref, C2V):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(C2V))

def get_6points_from_8points(eight_points):
    # eight points is nx3
    x_min = np.min(eight_points[:, 0])
    y_min = np.min(eight_points[:, 1])
    z_min = np.min(eight_points[:, 2])
    x_max = np.max(eight_points[:, 0])
    y_max = np.max(eight_points[:, 1])
    z_max = np.max(eight_points[:, 2])

    six_points = [x_min, y_min, z_min, x_max, y_max, z_max]
    return six_points

if __name__ == '__main__':
    # test()
    count = 0
    no_count = 0
    for file in os.listdir("./data/KITTI/training/image_2"):
        filename = str(file[:-4])
        #proposals , gt_boxes = get_pixel_coordinates(filename)
        proposals, gt_boxes, proposals_3d, gt_boxes_3d = get_pixel_coordinates_and_3d(filename, isplot=True)
        print(filename, gt_boxes.shape, 
              proposals.shape, 
              gt_boxes_3d.shape,
              proposals_3d.shape,)
        print("Proposals minx", np.min(proposals[:, 0]),
              "Proposals miny", np.min(proposals[:, 1]),
              "Proposals maxx", np.max(proposals[:, 2]),
              "Proposals maxy", np.max(proposals[:, 3]))
        image_org = read_image("./data/" + "KITTI/training/image_2/{}.png".format(filename))
        image_test = Image.open("./data/" + "KITTI/training/image_2/{}.png".format(filename))
        
        image = cv2.resize(image_org, (512, 512))
        count += 1 if gt_boxes.shape[0] == 0 else 0    
        no_count += 1 if gt_boxes.shape[0] != 0 else 0
        print(image_org.shape)
        if proposals.shape[0] != 0:
            proposals[:, 0] = proposals[:, 0] * (512 / image_test.size[0])
            proposals[:, 1] = proposals[:, 1] * (512 / image_test.size[1])
            proposals[:, 2] = proposals[:, 2] * (512 / image_test.size[0])
            proposals[:, 3] = proposals[:, 3] * (512 / image_test.size[1])

        if gt_boxes.shape[0] != 0:
            gt_boxes[:, 0] = gt_boxes[:, 0] * (512 / image_test.size[0])
            gt_boxes[:, 1] = gt_boxes[:, 1] * (512 / image_test.size[1])
            gt_boxes[:, 2] = gt_boxes[:, 2] * (512 / image_test.size[0])
            gt_boxes[:, 3] = gt_boxes[:, 3] * (512 / image_test.size[1])

        img = plot_2d_boxes(image, proposals, color=(0,0,255))
        #img = plot_2d_boxes(img, boxes2, color=(255,0,0))
        img = plot_2d_boxes(image, gt_boxes, color=(0,255,0))

        # cv2.imshow("image_{}".format(0), img)
        # print(image_org.shape)
        # print(image_test.size)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print("Zero gt ", count, "Non count ", no_count)