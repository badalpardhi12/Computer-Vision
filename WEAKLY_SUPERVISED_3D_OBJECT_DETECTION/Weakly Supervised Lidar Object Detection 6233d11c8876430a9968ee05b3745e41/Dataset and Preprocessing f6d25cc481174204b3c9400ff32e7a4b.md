# Dataset and Preprocessing

In this project we have used the publicly available KITTI dataset to train and evaluate our model. The label file of a data point in the dataset contains information such as class name, truncation, occlusion, 2D bounding boxes, dimensions of 3D bounding boxes, location of 3D bounding box, and heading angle. We only use the semantic class name of as the input label to our network. However, for evaluation and comparison purposes we use the 3D bounding box coordinates label. 

The tricky part here is to apply transformations that convert coordinates provided in camera coordinates to lidar coordinates and vice versa using the projection and calibration matrices.

# 1) Proposal Generation:

We have used PCL and Open3D libraries that perform KD-tree and DB-Scan clustering on all the 7481 velodyne lidar data samples in the KITTI training set. Every velodyne data point has a list of 3D proposals associated with it. We use the clustering based approach because of its computation speed (<1 ms) to generate clusters present in a given dense point cloud.

The proposals generated from the above mentioned approaches have 6 parameters associated with them *[x_min, y_min, z_min, x_max, y_max, z_max].* These parameters are in the velodyne or lidar coordinates which need to be converted to camera coordinates using the calibration matrices provided in the KITTI dataset.

After converting the proposals to camera coordinates, we then project these on the image by converting them into 2D bounding boxes. We use the projection matrix mentioned in the KITTI dataset calib file for each data point to get the bounding box values (in pixels).

![023DA2D6-C3B6-493D-A048-1912B5883195.png](Dataset%20and%20Preprocessing%20f6d25cc481174204b3c9400ff32e7a4b/023DA2D6-C3B6-493D-A048-1912B5883195.png)

![4B614AE8-AA41-4FFF-ADE0-2DE632925FA1.png](Dataset%20and%20Preprocessing%20f6d25cc481174204b3c9400ff32e7a4b/4B614AE8-AA41-4FFF-ADE0-2DE632925FA1.png)

![CF892C49-A362-44C5-97D3-36D858ACF504.png](Dataset%20and%20Preprocessing%20f6d25cc481174204b3c9400ff32e7a4b/CF892C49-A362-44C5-97D3-36D858ACF504.png)

Shown above are some examples of projected proposals that are generated from clustering based approaches. The red bounding boxes correspond to KD-tree based clustering, while the blue bounding boxes correspond to DB-Scan based clustering with appropriate threshold. The green boxes are the labelled ground truth. It can be seen that there are more number of proposals in the regions where the ground truth is present. This might prove helpful while performing ROI pooling that is explained in further section.

# 2) Data Augmentation and FOV cropping:

We augment the data by randomly shifting the horizontal and vertical axis of the proposals and also by randomly changing the size of these boxes with a small amount. The shifting constant in both the cases is experimental.

Since the KITTI dataset only has labels associated to the camera field of view, we try to restrict the use of our generated proposals and detections in the camera FOV space as well. This includes cropping the detected proposals that are out of the image field of view.

# 3) Alternate explored approaches:

Before finalising on the image based learning for object detection, we tried to follow [PIXOR’s](https://arxiv.org/abs/1902.06326) approach, that is, to use the stacked Bird’s Eye View (BEV) of the lidar point cloud at different depths as an input.

Below is the proposal generation for this approach:

[trim.4FB14269-B4EB-429D-A4FE-055687B49C28.MOV](Dataset%20and%20Preprocessing%20f6d25cc481174204b3c9400ff32e7a4b/trim.4FB14269-B4EB-429D-A4FE-055687B49C28.mov)

The proposals generated from 3D clustering algorithms in this case are projected on the BEV of the image and then fed to the PIXOR’s backbone network.