# Petrographic-Image-Segmentation
Project aimed to segment petrographic images taken from rock samples thin sections, in order to classify rock types.

The project will use an U-net segmentation architecture (https://arxiv.org/abs/1505.04597) implemented in Tensorflow Keras.

![Unet Architecture](images/u-net-architecture.png)

The idea is to start first with simple sandstone images to segment the principal minerals and porosity as a proof of concept and then continue to add segmentation of more sedimentary features to the model.


![petro image1](images/image1.png)

The following are examples of training images with 256x256 size.
First is the parallel nicols image second is the crossed nicols and last is the annotated mask.

![sanity1](images/sanity_check_1.jpg)

![sanity2](images/sanity_check_2.jpg)

![sanity3](images/sanity_check_3.jpg)
