# 3D Point Cloud Get Started

This summary includes traditional algorithms and deep learning methods, which refers to [paper with code](https://paperswithcode.com/), [shenlanxueyuan](https://www.shenlanxueyuan.com/my/course/204), and [My blog](https://blog.csdn.net/taifengzikai) (in Chinese). 

I participated in the LIDAR object detection track of [DEECAMP 2020](https://deecamp.com/2020/). The added information refers to the baseline of this competition track.

# Lidar and point cloud

For the principle of lidar, you can refer to the following articles:

* https://zhuanlan.zhihu.com/p/33792450
* https://pdal.io/workshop/lidar-introduction.html

# Dataset

- **PASCAL3D+ (2014)** [[Link\]](http://cvgl.stanford.edu/projects/pascal3d.html) 
  12 categories, on average 3k+ objects per category, for 3D object detection and pose estimation.

- **ModelNet (2015)** [[Link\]](http://modelnet.cs.princeton.edu/#) 
  127915 3D CAD models from 662 categories 
  ModelNet10: 4899 models from 10 categories 
  ModelNet40: 12311 models from 40 categories, all are uniformly orientated

- **ShapeNet (2015)** [[Link\]](https://www.shapenet.org/) 
  3Million+ models and 4K+ categories. A dataset that is large in scale, well organized and richly annotated. 
  ShapeNetCore [[Link\]](http://shapenet.cs.stanford.edu/shrec16/): 51300 models for 55 categories.

- **NYU Depth Dataset V2 (2012)** [[Link\]](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) 
  1449 densely labeled pairs of aligned RGB and depth images from Kinect video sequences for a variety of indoor scenes.

- **SUNRGB-D 3D Object Detection Challenge** [[Link\]](http://rgbd.cs.princeton.edu/challenge.html) 
  19 object categories for predicting a 3D bounding box in real world dimension 
  Training set: 10,355 RGB-D scene images, Testing set: 2860 RGB-D images

- **ScanNet (2017)** [[Link\]](http://www.scan-net.org/) 
  An RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations.

- **Facebook House3D: A Rich and Realistic 3D Environment (2017)** [[Link\]](https://github.com/facebookresearch/House3D) 
  House3D is a virtual 3D environment which consists of 45K indoor scenes equipped with a diverse set of scene types, layouts and objects sourced from the SUNCG dataset. All 3D objects are fully annotated with category labels. Agents in the environment have access to observations of multiple modalities, including RGB images, depth, segmentation masks and top-down 2D map views.

- **KITTI Benckmark**
    [paper link](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)
    The [KITTI](http://www.cvlibs.net/datasets/kitti/) (**K**arlsruhe **I**nstitute of **T**echnology and **T**oyota Technological **I**nstitute) dataset is a widely used computer vision benchmark which was released in 2012. A Volkswagen station was fitted with grayscale and color cameras, a Velodyne 3D Laser Scanner and a GPS/IMU system. They have datasets for various scenarios like urban, [residential](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=residential), [highway](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=road), and [campus](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=campus). 

- **nuScenes Benckmark**
    [paper link](https://arxiv.org/abs/1903.11027v1)
    nuTonomy scenes (nuScenes) is the first dataset to carry the full autonomous vehicle sensor suite: 6 cameras, 5 radars and 1 lidar, all with full 360 degree field ofview. nuScenes comprises 1000 scenes, each 20s long and fully annotated with 3D bound- ing boxes for 23 classes and 8 attributes. It has 7x as many annotations and 100x as many images as the pioneering KITTI dataset.

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

PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation

 [\[Paper\]](http://arxiv.org/abs/1807.00652)

# Point Cloud Augmentation
PointAugment: an Auto-Augmentation Framework for Point Cloud Classification

 [\[Paper\]](https://arxiv.org/pdf/2002.10876v2.pdf) [\[Code\]](https://github.com/liruihui/PointAugment) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/105183225)

# 3D Shape Completion
Learning Representations and Generative Models for 3D Point Clouds

 [\[Paper\]](https://arxiv.org/abs/1707.02392) [\[Code\]](https://github.com/optas/latent_3d_points) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/100812608)

# 3D Object Detection

Second: Sparsely embedded convolutional detection

 [\[Paper\]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6210968/)

VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

 [\[Paper\]](https://arxiv.org/pdf/1711.06396v1.pdf) [\[Code\]](https://github.com/qianguih/voxelnet)

PIXOR: Real-time 3D Object Detection from Point Clouds

 [\[Paper\]](https://arxiv.org/abs/1902.06326v3)

PointPillars: Fast Encoders for Object Detection from Point Clouds

 [\[Paper\]](https://arxiv.org/pdf/1812.05784v2.pdf)  [\[Code\]](https://github.com/nutonomy/second.pytorch)

PointRCNN：3D Object Proposal Generation and Detection from Point Cloud

 [\[Paper\]](https://arxiv.org/abs/1812.04244) [\[Code\]](https://github.com/sshaoshuai/PointRCNN)  [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/96840993)

IPOD: Intensive Point-based Object Detector for Point Cloud

 [\[Paper\]](http://arxiv.org/abs/1812.05276)

Deep Hough Voting for 3D Object Detection in Point Clouds

 [\[Paper\]](http://arxiv.org/abs/1904.09664)

YOLO3D

 [\[Paper\]](https://arxiv.org/pdf/1808.02350v1.pdf)

Complex-YOLO: An Euler-Region-Proposal for Real-time 3D Object Detection on Point Clouds

 [\[Paper\]](https://arxiv.org/abs/1803.06199v2)

Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds

 [\[Paper\]](https://arxiv.org/abs/1904.07537v1)

Joint 3D Proposal Generation and Object Detection from View Aggregation

 [\[Paper\]](https://arxiv.org/abs/1712.02294v4)

Multi-View 3D Object Detection Network for Autonomous Driving

 [\[Paper\]](https://arxiv.org/abs/1611.07759)

PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection

 [\[Paper\]](https://arxiv.org/pdf/1912.13192.pdf) [\[Code\]](https://blog.csdn.net/taifengzikai/article/details/106480264) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/106480264)

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

# Blogs

- https://zhuanlan.zhihu.com/p/58734240

# reference

- [\[paper with code\]](https://paperswithcode.com/)
- [\[shenlanxueyuan\]](https://www.shenlanxueyuan.com/my/course/204)
- [\[My blog (in Chinese) \]](https://blog.csdn.net/taifengzikai) 
- [\[DEECAMP 2020\]](https://deecamp.com/2020/)
