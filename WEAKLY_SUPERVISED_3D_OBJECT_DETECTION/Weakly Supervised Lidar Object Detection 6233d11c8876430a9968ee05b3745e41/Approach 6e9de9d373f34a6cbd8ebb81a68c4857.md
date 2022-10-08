# Approach

## Methodology

Our Overall pipeline involves two main steps

1. Generating 3D object proposals
2. Object Detection using Neural Network

### Proposal Generation

Proposals mean locations in the point cloud where an object of interest can be present. Methods like selective search are very common in images. Hence there are plenty of weakly supervised object detection methods on images. We use Density-Based Clustering to cluster the point cloud to various objects and select the clusters which satisfy a certain density threshold. Before clustering, we segment out the ground plane to avoid interactions with objects of interest. The main parameters to control the quality and quantity of generated clusters are 

1. min number of points in a cluster (m)
2. maximum distance between two points in a cluster ($\epsilon$ )

We use n = 30,  and ϵ = 0.9 for our proposal generation.                 ﻿

![Clustered Segments](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/Screenshot_from_2022-05-09_04-23-02.png)

Clustered Segments

![Proposals from clusters](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/Screenshot_from_2022-05-09_04-23-51.png)

Proposals from clusters

---

![Clustered Segments](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/Screenshot_from_2022-05-09_04-21-24.png)

Clustered Segments

![Proposals from cluster](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/Screenshot_from_2022-05-09_04-21-51.png)

Proposals from cluster

We get on average, 20 - 30 proposals from a Lidar scene. Due to the accuracy loss of Lidar with increasing distance, we crop the scene along x from +/-35 m  to +/-25 m and along the y-axis from +80 to +40. For training, we consider only car class as the most crucial class for KITTI object detection.

### Object Detection using Neural Network

We tried two different approaches of which one produced negative results

1. Bird’s eye view based Object detection
2. Front Camera-Based Object detection
    
    ### Bird’s eye view based Object detection
    
    The workflow of BEV proposal generation is shown below. We project the 3d bounding box proposals from above to BEV, and we randomly augment the boxes to get a higher number of boxes. This produced negative results for us, which are discussed in “**Negative Results.”**
    
    ![The overall pipeline of proposal generation for Bev](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/Screenshot_from_2022-05-09_05-11-17.png)
    
    The overall pipeline of proposal generation for Bev
    
    We used the main pipeline from [WSDDN](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16/bilen16.pdf), where we replaced the encoder with [PIXOR](https://arxiv.org/abs/1902.06326) backbone pre-trained on the KITTI dataset for  BEV object detection. The network architecture is shown below. BCE loss is used to optimize the model.
    
    ![                                                                                                                                 **Network architecture BEV Weakly supervised detection.**](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/Screenshot_from_2022-05-09_05-08-11.png)
    
                                                                                                                                     **Network architecture BEV Weakly supervised detection.**
    
     
    
    ### Front Camera-Based Object detection
    
    As the BEV weakly supervised detection produced negative results we decided to move forward by doing the detection on Front View Images. Doing some geometric transformations based on camera calibration matrices we projected our proposals to the Front camera image of the KITTI dataset. The proposal generation Pipeline showed below.   
    
    ![The overall pipeline of proposal generation for Front camera](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/Screenshot_from_2022-05-09_18-17-06.png)
    
    The overall pipeline of proposal generation for Front camera
    
    For Front Camera-based object detection we used [Resnet-152](https://arxiv.org/abs/1512.03385) encoder until the third last layer, followed by ROI align and Fully connected layers for classification and detection.
    
    ![The architecture of Weakly-Supervised detection from front view camera images](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/Screenshot_from_2022-05-09_18-28-40.png)
    
    The architecture of Weakly-Supervised detection from front view camera images
    
    Input Image is passed through the Resnet-152 pre-trained network producing higher-level features. ROI Align is performed on the encoded features to extract ROIs. The ROIs are flattened and passed through the classifier to obtain features N*81920 (where N is the number of proposals, calculations shown below). 
    
    The detection and Recognition heads produce outputs of size N*num_classes (num_classes=1, only cars). Softmax is applied along the zeroth dimension of the detection head, and Sigmoid is applied along the first dimension of the recognition head. Softmax cannot be applied because as we have only 1 class it will make all the outputs 1 irrespective of network predictions till that point.  Both N*1 tensors are multiplied to form the output of dimension N*1. For calculating loss the output tensor is summed across the zeroth dimension. As the values are between 0 and 1 due to individual sigmoid and softmax in the previous step, no sigmoid is applied in the calculation of Binary Cross-Entropy Loss and Focal Loss.
    
     A dynamic threshold is applied to select a number of ROIs. Unlike selective search, we cannot get a large number of quality proposals from sparse lidar point clouds. So we get a varying number of proposals for each data sample. We design a dynamic threshold of 0.5 / N. Only ROIs above this threshold are passed to the next step of Non-Maxima Suppression. We use an IoU threshold of 0.2 in NMS. This might seem low, but KITTI images have a lot of overlapping ground truth boxes due to a large number of closely parked cars in some scenarios.
    
    ![Loss calculation and evaluation pipeline](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/Screenshot_from_2022-05-09_19-23-00.png)
    
    Loss calculation and evaluation pipeline
    
    ### Implementation Details
    
    Network Specifications
    
    1. Input Image: 3, 1242, 375
    2. After Resnet Encoder: 1024, 77, 24
    3. Roi Align size: (16, 5)
    4. Classifier, Recognition, and detection structure same as AlexNet
        1. Classifier: 81920 (1024 * 16 * 5) → 4096, 4096→4096
        2. Recognition head 4096 → 1
        3. Detection head 4096 → 1
    
    Training Details
    
    We train with Focal loss, for 10 epochs with an LR of SGD=0.0001, and a Step LR scheduler that reduces LR by a factor of 0.1 every 5 epochs. We train on g4dn.xlarge instance. We split the KITTI data as 5985 training data and 1496 testing data. 
    
    **ROI Align**
    
    ROI Align has the same goal as ROI Pooling. If the ratio of the original image (x) to ROI pool is k, ROI pooling divides the original proposal coordinates by k and takes the integer part as the new proposal coordinates. Then around the new region, max or average pooling is done. On the other hand ROI, Align keeps the new proposal coordinates as float values instead of discretizing them as integer values. Binary interpolation is used to get values in overlapping grid points. ROI alignment is preferred in situations where we need precise ROIs. As we have many overlapping or close ground truths and proposals we employ ROI align to get more accurate results.
    
    **Focal Loss**
    
    We trained our model on both BCE loss and [Focal loss](https://arxiv.org/abs/1708.02002), where focal loss seems to produce a high mAP. The focal loss was introduced to mitigate the problem of class imbalance. If BCE Loss is used network becomes biased towards the majority labels’ class, in this case, will tend to predict more cars. 
    
    ![focal.png](Approach%206e9de9d373f34a6cbd8ebb81a68c4857/focal.png)
    
    Focal loss introduces 2 hyperparameters to the CE loss, $\alpha$  , and $\gamma$. α accounts for the class imbalance, while $\gamma$  gives hard classes that are misclassified more weightage while correctly predicted easy classes a low weightage. These modifications affect the model training as we will see in the results