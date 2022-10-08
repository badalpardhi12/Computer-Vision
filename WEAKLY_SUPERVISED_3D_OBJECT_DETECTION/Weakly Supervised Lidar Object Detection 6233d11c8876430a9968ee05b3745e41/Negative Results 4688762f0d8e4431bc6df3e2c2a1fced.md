# Negative Results

As suggested in Approach our first model was a failure, we used our BEV-based weakly supervised detection model to train the data. We got mAP values close to zero.

| IoU | 2D BEV car AP |
| --- | --- |
| 0.1 | 5.11 |
| 0.2 | 2.1 |
| 0.3 | .79 |
| 0.4 | .32 |

![Screenshot from 2022-05-09 23-43-27.png](Negative%20Results%204688762f0d8e4431bc6df3e2c2a1fced/Screenshot_from_2022-05-09_23-43-27.png)

![Screenshot from 2022-05-09 23-43-29.png](Negative%20Results%204688762f0d8e4431bc6df3e2c2a1fced/Screenshot_from_2022-05-09_23-43-29.png)

                                    Predicted Boxes                                                                                                                          Ground truth car boxes

### Reasons for failure

1. **No high-quality starting weights:** Unlike normal RGB-based object detection, we cannot use ImageNet pre-trained weights on BEV images, because BEV images are very sparse compared to RGB images and have more channels. For instance, without loss of much information, our BEV contains 36 channels. 
2. **Occlusion of points:** Assuming the car faces in the y-direction, the backside of most of the objects of interest like other cars, pedestrians, etc are occluded. So the length of the proposal will be really inaccurate. When working on BEV this occlusion really affects ROI pooling, so the network fails to learn from ROIs.
3. **Weak feature Extractor (PIXOR):** For our BEV-based object detection we used PIXOR’s backbone as the feature extractor. PIXOR is trained for supervised object detection and contains a bounding box regressor.  PIXOR is kind of a single shot detection, so ROI pooling from encode feature maps produced weak features