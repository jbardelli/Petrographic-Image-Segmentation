# Petrographic-Image-Segmentation
Project is aimed at segmenting petrographic images taken from rock samples thin sections, in order to classify rock types.
The inital objective is to classifiy 4 different minerals that allow the sandstone classification in the Folk chart.
The classes are:
  1) Feldspar
  2) Lithic fragments
  3) Plagioclase
  4) Quartz

The project will compare results using U-net and DeepLab V3 models.

The U-net semantic segmentation architecture (https://arxiv.org/abs/1505.04597) will be implemented in Tensorflow Keras.

<img src="images/u-net-architecture.png" width="450" height="300">

The DeepLab model is based in the following paper (https://arxiv.org/abs/1706.05587), here I use the tensorflow implementation of DeepLab (available at https://github.com/tensorflow/models/tree/master/research/deeplab)

The idea is to start first with simple sandstone images to segment the principal minerals and porosity as a proof of concept and then continue to add segmentation of more sedimentary features to the model.

The following are examples of training images divided in 256x256 size.
First is the parallel polarized light (PPL) image, second is the crossed polarized light (XPL) image and last is the annotated mask. The two images as well as the mask have a pixel-to-pixel correspondence, and give more information as XPL shows other aspects of each mineral.
<img src="images/sanity_check_1.jpg" width="800" height="300">
<img src="images/sanity_check_2.jpg" width="810" height="300">
<img src="images/sanity_check_3.jpg" width="810" height="300">

In the U-net model I combine the two images (PPL and XPL) into a 6 channel image and adapted the input layer of the U-net model to accomodate to it. It is unclear yet if the Deeplab model can accept to images for the same rock section with different light.

Example of initial test with predictions using U-net with a model trained with 6 images.
<img src="images/prediction1.png" width="550" height="400">
Example of initial test with predictions using U-net with a model trained with 20 images.
<img src="images/prediction3.png" width="550" height="400">

