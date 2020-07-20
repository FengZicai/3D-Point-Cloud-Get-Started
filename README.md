# Different tasks for 3D Point Cloud

This summary includes traditional algorithms and deep learning methods, which refers to [paper with code](https://paperswithcode.com/), [shenlanxueyuan](https://www.shenlanxueyuan.com/my/course/204), and [My blog](https://blog.csdn.net/taifengzikai) (in Chinese).


# Tradition algorithms

The following algorithms and registration-related deep learning methods refer to the course named 3D Point Cloud Analysis. The code of which can be access [here](https://github.com/AlexGeControl/3D-Point-Cloud-Analytics).

PCA & Kernel PCA

Suface Normal

DownSampling

- grid sampling
- FPS
- Normal Space Sampling

Nearset Neighbor Problem

- KD-Tree
- Octree

Clustering

- k-means
- GMM
- EM
- Spectral Clustering
- Mean Shift
- DBSCAN

Model Fitting

- Least squares
- Hough Transformation
- RANSAC

Feature Detection

- Harris 3D&6D
- ISS（Intrinsic Shape Signatures）

Feature Description

- PFH & FPFH
- SHOT

Registration

- ICP
- NDT



# Review article
Deep Learning for 3D Point Clouds: A Survey

 [\[Paper\]](https://arxiv.org/abs/1912.12033) [\[Code\]](https://github.com/QingyongHu/SoTA-Point-Cloud) [\[My Blog1\]](https://blog.csdn.net/taifengzikai/article/details/104109562) [\[My Blog2\]](https://blog.csdn.net/taifengzikai/article/details/104153717)

Review: deep learning on 3D point clouds

 [\[Paper\]](https://arxiv.org/abs/2001.06280v1)

A Review on Object Pose Recovery: from 3D Bounding Box Detectors to Full 6D Pose Estimators

 [\[Paper\]](https://arxiv.org/abs/2001.10609v1)

# 3D Point Cloud Classification and Segmentation
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

 [\[Paper\]](https://arxiv.org/pdf/1612.00593v2.pdf) [\[Code\]](https://github.com/charlesq34/pointnet) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/104438739)

PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

 [\[Paper\]](https://arxiv.org/pdf/1706.02413v1.pdf) [\[Code\]](https://github.com/charlesq34/pointnet2) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/104438739)

PointConv: Deep Convolutional Networks on 3D Point Clouds

 [\[Paper\]](https://arxiv.org/abs/1811.07246) [\[Code\]](https://github.com/DylanWusee/pointconv) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/90897467)

# Point Cloud Augmentation
PointAugment: an Auto-Augmentation Framework for Point Cloud Classification

 [\[Paper\]](https://arxiv.org/pdf/2002.10876v2.pdf) [\[Code\]](https://github.com/liruihui/PointAugment) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/105183225)

# 3D Shape Completion
Learning Representations and Generative Models for 3D Point Clouds

 [\[Paper\]](https://arxiv.org/abs/1707.02392) [\[Code\]](https://github.com/optas/latent_3d_points) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/100812608)

# 3D Object Detection
VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

 [\[Paper\]](https://arxiv.org/pdf/1711.06396v1.pdf) [\[Code\]](https://github.com/qianguih/voxelnet)

PointRCNN：3D Object Proposal Generation and Detection from Point Cloud

 [\[Paper\]](https://arxiv.org/abs/1812.04244) [\[Code\]](https://github.com/sshaoshuai/PointRCNN)  [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/96840993)

PointPillars: Fast Encoders for Object Detection from Point Clouds

 [\[Paper\]](https://arxiv.org/pdf/1812.05784v2.pdf)  [\[Code\]](https://github.com/nutonomy/second.pytorch)

PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection

 [\[Paper\]](https://arxiv.org/pdf/1912.13192.pdf) [\[Code\]](https://blog.csdn.net/taifengzikai/article/details/106480264)

PointPainting: Sequential Fusion for 3D Object Detection

 [\[Paper\]](https://arxiv.org/pdf/1911.10150v2.pdf) [\[Code\]](https://github.com/rshilliday/painting) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/106520603)

# Moving Point Cloud Processing
PointRNN: Point Recurrent Neural Network for Moving Point Cloud Processing

 [\[Paper\]](https://arxiv.org/pdf/1910.08287.pdf) [\[Code1\]](https://github.com/hehefan/PointRNN) [\[Code2\]](https://github.com/hehefan/PointRNN-PyTorch) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/107129194)
# 3D Object Tracking 
Leveraging Shape Completion for 3D Siamese Tracking

 [\[Paper\]](ttps://arxiv.org/abs/1903.01784) [\[Code\]](https://github.com/SilvioGiancola/ShapeCompletion3DTracking)

P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds

 [\[Paper\]](https://arxiv.org/pdf/2005.13888v1.pdf) [\[Code\]](https://github.com/HaozheQi/P2B)

# 3D Multi-Object Tracking
A Baseline for 3D Multi-Object Tracking

 [\[Paper\]](https://arxiv.org/pdf/1907.03961v4.pdf) [\[Code\]](https://github.com/xinshuoweng/AB3DMOT)

Probabilistic 3D Multi-Object Tracking for Autonomous Driving

 [\[Paper\]](https://arxiv.org/pdf/2001.05673v1.pdf) [\[Code\]](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking)

Center-based 3D Object Detection and Tracking

 [\[Paper\]](https://arxiv.org/pdf/2006.11275v1.pdf) [\[Code\]](https://github.com/tianweiy/CenterPoint)

Argoverse: 3D Tracking and Forecasting With Rich Maps

 [\[Paper\]](https://arxiv.org/abs/1911.02620) [\[Code\]](https://github.com/argoai/argoverse-api) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/104096337)

# Feature Detection & Feature Description & Registration
3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration

 [\[Paper\]](https://arxiv.org/pdf/1807.09413v1.pdf) [\[Code\]](https://github.com/yewzijian/3DFeatNet)

SO-Net: Self-Organizing Network for Point Cloud Analysis

 [\[Paper\]](https://arxiv.org/pdf/1803.04249v4.pdf) [\[Code\]](https://github.com/lijx10/SO-Net)


USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds
 
 [\[Paper\]](https://arxiv.org/pdf/1904.00229v1.pdf) [\[Code\]](
 https://github.com/lijx10/USIP)

 

# reference

- [\[paper with code\]](https://paperswithcode.com/)
- [\[shenlanxueyuan\]](https://www.shenlanxueyuan.com/my/course/204)
- [\[My blog (in Chinese) \]](https://blog.csdn.net/taifengzikai) 
