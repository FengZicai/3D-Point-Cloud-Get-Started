# 3D Point Cloud Get Started

This summary includes traditional algorithms and deep learning methods, which refers to [paper with code](https://paperswithcode.com/), [shenlanxueyuan](https://www.shenlanxueyuan.com/my/course/204), and [My blog](https://blog.csdn.net/taifengzikai) (in Chinese). 

<!-- I participated in the LIDAR object detection track of [DEECAMP 2020](https://deecamp.com/2020/). The added information refers to the baseline of this competition track.
 -->

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

- **ScanObjectNN (2019)** [[Link\]](https://github.com/hkust-vgd/scanobjectnn) is built upon SceneNN and ScanNet, contains 
  15,000 objects from 15 different classes, and presents significant challenges. 


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
    [[Link\]](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)
    The [KITTI](http://www.cvlibs.net/datasets/kitti/) (**K**arlsruhe **I**nstitute of **T**echnology and **T**oyota Technological **I**nstitute) dataset is a widely used computer vision benchmark which was released in 2012. A Volkswagen station was fitted with grayscale and color cameras, a Velodyne 3D Laser Scanner and a GPS/IMU system. They have datasets for various scenarios like urban, [residential](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=residential), [highway](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=road), and [campus](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=campus). 

- **nuScenes Benckmark**
    [[Link\]](https://arxiv.org/abs/1903.11027v1)
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



# Deep Learning Methods
Deep Learning for 3D Point Clouds: A Survey

 [\[Paper\]](https://arxiv.org/abs/1912.12033) [\[Code\]](https://github.com/QingyongHu/SoTA-Point-Cloud) [\[My Blog1\]](https://blog.csdn.net/taifengzikai/article/details/104109562) [\[My Blog2\]](https://blog.csdn.net/taifengzikai/article/details/104153717)

Review: deep learning on 3D point clouds

 [\[Paper\]](https://arxiv.org/abs/2001.06280v1)

A Review on Object Pose Recovery: from 3D Bounding Box Detectors to Full 6D Pose Estimators

 [\[Paper\]](https://arxiv.org/abs/2001.10609v1)

# 3D Object Classification
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

 [\[Paper\]](https://arxiv.org/pdf/1612.00593v2.pdf) [\[Code\]](https://github.com/charlesq34/pointnet) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/104438739)

PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

 [\[Paper\]](https://arxiv.org/pdf/1706.02413v1.pdf) [\[Code\]](https://github.com/charlesq34/pointnet2) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/104438739)

PointConv: Deep Convolutional Networks on 3D Point Clouds

 [\[Paper\]](https://arxiv.org/abs/1811.07246) [\[Code\]](https://github.com/DylanWusee/pointconv) [\[My Blog\]](https://blog.csdn.net/taifengzikai/article/details/90897467)

PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation

 [\[Paper\]](http://arxiv.org/abs/1807.00652) [\[Code\]](https://github.com/MVIG-SJTU/pointSIFT)

Dynamic Graph CNN for Learning on Point Clouds

[\[Paper\]](https://arxiv.org/abs/1801.07829) [\[Code\]](https://github.com/WangYueFt/dgcnn)

Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling

[\[Paper\]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Mining_Point_Cloud_CVPR_2018_paper.pdf) [\[Code\]](http://www.merl.com/research/license#KCNet)

PointCNN: Convolution On X-Transformed Points

[\[Paper\]](https://papers.nips.cc/paper_files/paper/2018/hash/f5f8590cd58a54e94377e6ae2eded4d9-Abstract.html) [\[Code\]](https://github.com/yangyanli/PointCNN)

Relation-Shape Convolutional Neural Network for Point Cloud Analysis

[\[Paper\]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Relation-Shape_Convolutional_Neural_Network_for_Point_Cloud_Analysis_CVPR_2019_paper.pdf) [\[Code\]](https://github.com/Yochengliu/Relation-Shape-CNN)

DensePoint: Learning Densely Contextual Representation for Efficient Point Cloud Processing

[\[Paper\]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_DensePoint_Learning_Densely_Contextual_Representation_for_Efficient_Point_Cloud_Processing_ICCV_2019_paper.pdf) [\[Code\]](https://github.com/Yochengliu/DensePoint)


Geometric Back-projection Network for Point Cloud Classification

[\[Paper\]](https://arxiv.org/abs/1911.12885) [\[Code\]](https://github.com/ShiQiu0419/GBNet)

Point Transformer

[\[Paper\]](https://arxiv.org/abs/2011.00931) [\[Code\]](https://github.com/engelnico/point-transformer)

PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling

[\[Paper\]](https://arxiv.org/abs/2003.00492) [\[Code\]](https://github.com/yanx27/PointASNL)

A Closer Look at Local Aggregation Operators in Point Cloud Analysis

[\[Paper\]](https://arxiv.org/abs/2007.01294) [\[Code\]](https://github.com/zeliu98/CloserLook3D)


Learning Geometry-Disentangled Representation for Complementary Understanding of 3D Object Point Cloud

[\[Paper\]](https://arxiv.org/abs/2012.10921) [\[Code\]](https://github.com/mutianxu/GDANet)

MVT: Multi-view Vision Transformer for 3D Object Recognition

[\[Paper\]](https://arxiv.org/abs/2110.13083) [\[Code\]](https://github.com/shanshuo/MVT)


PCT: Point Cloud Transformer

[\[Paper\]](https://arxiv.org/pdf/2012.09688.pdf) [\[Code\]](https://github.com/MenghaoGuo/PCT)

PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds

[\[Paper\]](https://arxiv.org/abs/2103.14635) [\[Code\]](https://github.com/CVMI-Lab/PAConv)


MVTN: Multi-View Transformation Network for 3D Shape Recognition

[\[Paper\]](https://arxiv.org/abs/2011.13244) [\[Code\]](https://github.com/ajhamdi/MVTN)


Point Transformer

[\[Paper\]](https://arxiv.org/abs/2012.09164) [\[Code\]](https://github.com/POSTECH-CVLab/point-transformer)


Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis

[\[Paper\]](https://arxiv.org/abs/2105.01288) [\[Code\]](https://github.com/tiangexiang/CurveNet)


POINTVIEW-GCN: 3D SHAPE CLASSIFICATION WITH MULTI-VIEW POINT CLOUDS

[\[Paper\]](https://ieeexplore.ieee.org/document/9506426) [\[Code\]](https://github.com/SMohammadi89/PointView-GCN)


REVISITING POINT CLOUD CLASSIFICATION WITH A SIMPLE AND EFFECTIVE BASELINE

[\[Paper\]](http://proceedings.mlr.press/v139/goyal21a/goyal21a.pdf) [\[Code\]](https://github.com/princeton-vl/SimpleView)


Multi-View 3D Shape Recognition via Correspondence-Aware Deep Learning

[\[Paper\]](https://ieeexplore.ieee.org/iel7/83/4358840/09442303.pdf)

PRA-Net: Point Relation-Aware Network for 3D Point Cloud Analysis

[\[Paper\]](https://arxiv.org/abs/2112.04903) [\[Code\]](https://github.com/XiwuChen/PRA-Net)


Dense-Resolution Network for Point Cloud Classification and Segmentation

[\[Paper\]](https://openaccess.thecvf.com/content/WACV2021/papers/Qiu_Dense-Resolution_Network_for_Point_Cloud_Classification_and_Segmentation_WACV_2021_paper.pdf) [\[Code\]](https://github.com/ShiQiu0419/DRNet)


APP-Net: Auxiliary-point-based Push and Pull Operations for Efficient Point Cloud Classification

[\[Paper\]](https://arxiv.org/abs/2205.00847) [\[Code\]](https://github.com/MCG-NJU/APP-Net)

Masked Autoencoders for Point Cloud Self-supervised Learning

[\[Paper\]](https://arxiv.org/abs/2203.06604) [\[Code\]](https://github.com/Pang-Yatian/Point-MAE)

PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies

[\[Paper\]](https://arxiv.org/abs/2206.04670) [\[Code\]](https://github.com/guochengqian/PointNeXt)

Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling

[\[Paper\]](https://arxiv.org/abs/2111.14819) [\[Code\]](https://github.com/lulutang0608/Point-BERT)

Surface Representation for Point Clouds

[\[Paper\]](https://arxiv.org/abs/2205.05740) [\[Code\]](https://github.com/hancyran/RepSurf)

RETHINKING NETWORK DESIGN AND LOCAL GEOMETRY IN POINT CLOUD: A SIMPLE RESIDUAL MLP FRAMEWORK

[\[Paper\]](https://arxiv.org/abs/2202.07123) [\[Code\]](https://github.com/ma-xu/pointMLP-pytorch)

Points to Patches: Enabling the Use of Self-Attention for 3D Shape Recognition

[\[Paper\]](https://arxiv.org/abs/2204.03957) [\[Code\]](https://github.com/axeber01/point-tnt)


# 3D Point Cloud Segmentation


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

 LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention
 
 [\[Code\]](https://github.com/yinjunbo/3DVID)

# 4D Semantic Segmentation

4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks

[\[Code\]](https://github.com/StanfordVL/MinkowskiEngine))

LiDAR-based Recurrent 3D Semantic Segmentation with Temporal Memory Alignment

SpSequenceNet: Semantic Segmentation Network on 4D Point Clouds

[\[Code\]](https://github.com/dante0shy/SpSequenceNet)

Dynamic Semantic Occupancy Mapping using 3D Scene Flow and Closed-Form Bayesian Inference

Spatial-Temporal Transformer for 3D Point Cloud Sequences

Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos

[\[Code\]](https://github.com/hehefan/P4Transformer)

TempNet: Online Semantic Segmentation on Large-scale Point Cloud Series

PSTNET: POINT SPATIO-TEMPORAL CONVOLUTION ON POINT CLOUD SEQUENCES

[\[Code\]](https://github.com/hehefan/Point-Spatio-Temporal-Convolution)


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
