# Medical-Image-Processing

Firstly, if you wish to get started with medical image processing, it is necessary to learn a machine learning language. Nowadays, the prevalent machine learning languages include TensorFlow, JAX, and Torch. We recommend Torch as your entry-level language here, due to its abundance of open-source projects and deployment tutorials.

Python: https://www.youtube.com/watch?v=MbhUGst0rqw&list=PLVyDH2ns1F75k1hvD2apA0DwI3XMiSDqp

Torch: https://www.youtube.com/watch?v=rDhBcP4ikpA&list=PLDzdzeKX7DWep2KyJwJ-BmYciXTARvZO6

Indeed, you do not need to learn every detail in Python and Torch; a general understanding of data manipulation therein is usually sufficient. You can refer to the API for queries as you use it moving forward.

Presently, medical image processing encompasses areas including segmentation, registration, molecular construction, and others. Here, we will provide a brief introduction to the content concerning segmentation. Current medical image segmentation falls generally into two major categories: optimization of medium and small-scale models based on directions such as UNet, SegFormer, and others. And another direction rooted in prompts based on the SAM model.

Firstly, the reference materials needed for the first direction are as follows (paper name and url):

UNet: [U-Net: Convolutional Networks for Biomedical Image Segmentation] https://arxiv.org/abs/1505.04597

UNet++: [UNet++: A Nested U-Net Architecture for Medical Image Segmentation] https://arxiv.org/abs/1807.10165

SegFormer: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers] https://arxiv.org/abs/2105.15203

PSPNet: [Pyramid Scene Parsing Network] https://arxiv.org/abs/1612.01105

HRnet: [Deep High-Resolution Representation Learning for Visual Recognition] https://arxiv.org/abs/1908.07919

DeepLab: [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs] https://arxiv.org/abs/1606.00915

UNeXt: [UNeXt: MLP-based Rapid Medical Image Segmentation Network] https://arxiv.org/abs/2203.04967

All of the aforementioned papers are outstanding achievements and warrant a thorough reading. Moreover, under necessary circumstances, search for their official implementations on GitHub and refer to their training processes. We have integrated these models into the model.py file for reference purposes only.
