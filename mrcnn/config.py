"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.
# 配置基类
# 不要直接使用这个类。继承该类并重写需要改变的配置属性

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    # 为这个配置命名。例如, 'COCO', 'Experiment 3', 等等。
    NAME = None  # Override in sub-classes 在子类中重写它

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    # 用于训练的GPU数量。如果用CPU训练，设为1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    # 每个GPU一次能处理的图像数量。以12G的GPU来说，可以处理2张1024*1024的图像
    # 根据你的GPU显存和图像大小调整它，使你的GPU的性能得到最佳利用
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    # 每个epoch训练的步数
    # 这个参数不需要匹配训练集的大小。在每个epoch训练结束后，
    # Tensorboard的更新将被保存，因此将其设置得较小时，更新tensorboard的
    # 频次更高。验证集的状态也将在每个epoch结束后更新，这需要一定的时间，
    # 所以不要讲这个数字设置得太小，避免大量时间浪费在验证上。
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    # 每个训练epoch之后，验证的步数，数字越大，验证的状态也准确，但是减慢训练。
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    # Backbone网络结构，支持resnet50和resnet101
    # 也可以提供一个构建网络图的回调函数给model.resnet_graph
    # 以及计算骨干每层大小的回调函数
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    # 当你提供了骨干网络时，需要配置一个回调函数用来计算FPN金字塔的层大小
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    # Resnet101的FPN金字塔中，每层对应的stride
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    # 分类网络中全连接层FC的大小
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    # 特征金字塔中，每层的通道数量
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    # 类别数量，包含背景
    NUM_CLASSES = 1  # Override in sub-classes，在子类中重写

    # Length of square anchor side in pixels
    # 以像素计算，每个正方形anchor的边长
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    # 每个cell中anchors的宽/高比
    # 1 代表正方形anchor，0.5代表宽为0.5的长方形
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    # anchor的步长，
    # 1，表示骨干网络特征图上的每个cell都产生anchors，
    # 2，每隔一个cell产生anchors，等等，以此类推。
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # RPN proposals的非极大值抑制（NMS）阈值
    # 你可以通过增大阈值来产生更多proposals
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    # 一张图上的anchors数量
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    # 经过top_k筛选后，用来参与NMS处理的ROIs数量
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    # 经过NMS处理后保留的ROIs数量
    # training 2000，推断（预测）1000
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    # 使用它，将会把实例mask缩放到更小的尺寸，以减小内存负载。
    # 当处理高分辨率图像时，推荐设置。
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    # 输入图像的缩放
    # 通常，使用‘正方形’缩放模式用于训练和预测，在大多数情景，都表现比较好。
    # 在这个模式中，缩放图像，使其短边=IMAGE_MIN_DIM，
    # 同时确保长边不大于IMAGE_MAX_DIM，其余地方用零填充。
    # 这样可以使图像对齐，方便一个batch可以处理多幅图像。
    # none:   无缩放或填充，返回原图。
    # square: 缩放或填充0，返回[max_dim, max_dim]大小的图像。
    # pad64:  宽和高填充0，使他们成为64的倍数。
    #         如果IMAGE_MIN_DIM 或 IMAGE_MIN_SCALE不为None, 则在填充之前先
    #         缩放。IMAGE_MAX_DIM在该模式中被忽略。
    #         要求为64的倍数是因为在对FPN金字塔的6个levels进行上/下采样时保证平滑(2**6=64).
    # crop:   对图像进行随机裁剪。首先, 基于IMAGE_MIN_DIM和IMAGE_MIN_SCALE
    #         对图像进行缩放, 然后随机裁剪IMAGE_MIN_DIM x IMAGE_MIN_DIM大小。
    #         仅在训练时使用。

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    # 最小缩放比例。在MIN_IMAGE_DIM后检测，能强制放大。
    # 比如，设置为2时，图像的宽高都将放大2倍，尽管不满足IMAGE_MAX_DIM
    # 然而, 在'square'模式中,该参数会被IMAGE_MAX_DIM覆盖。
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    # 图像的颜色通道。RGB = 3, grayscale = 1, RGB-D = 4
    # 改变通道值，需要改动其他代码，详情参考
    # https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    # RGB均值
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # 每张图喂给classifier/mask heads的ROIs数量
    # Mask RCNN论文中为512，但是RPN通常产生不了这么多正样本 proposals（目标框），
    # 所以这里保证positive:negative=1:3，可以通过增大RPN NMS阈值来获得更多proposals
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    # 训练classifier/mask heads时，positive ROIs的比例
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    # 池化ROI的尺寸
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    # 输出mask的大小
    # 更改这个参数需要修改mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    # 一张图中，最大gt实例数量
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    # RPN和最终检测的bbox 精修的标准差（有什么依据？需要仔细看paper）
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    # 最终检测时最大实例数量
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    # 实例被接受的最小概率值，小于的将被忽略
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    # 检测时NMS的阈值
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    # lr和动量
    # Mask RCNN paper中，lr=0.02，但TensorFlow框架下，会导致权重爆炸，
    # 可能因为优化器实现的差异导致。
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    # 正则化系数
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    # 更精确的优化Loss weights，可以用于R-CNN训练设置。
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    # 可以使用RPN ROIs或外部生成的ROIs用于训练。
    # 在大部分情况下将该值设为true。
    # 如果在训练 head branches时，不使用RPN生成的ROIs，将其设为false。
    # 例如, 调试classifier head而不不需要训练RPN时。
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    # 是否使用batch normalization layers
    #     None: 训练BN layers. 这是标准模式。
    #     False: 冻结BN layers. 在小的batch size时比较好。
    #     True: (未使用)，即使是在预测时也将layer设为训练模式。
    TRAIN_BN = False  # Defaulting to False since batch size is often small
    # 默认为False，因为batchs ize比较小。

    # Gradient norm clipping
    # 将梯度最大值限制在5.0
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """Set values of computed attributes."""
        # 设置计算属性值
        # Effective batch size
        # 有效batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        # 输入图像大小
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        # "image_id": image_id,
        # "original_image_shape": original_image_shape,
        # "image_shape": image_shape,
        # "window": window,
        # "scale": scale,
        # "active_class_ids": active_class_ids,
        # 图像meta data长度
        # 更多细节参见compose_image_meta()
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        # 显示配置属性
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
