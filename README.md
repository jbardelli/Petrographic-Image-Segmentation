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

The idea is to start first with simple sandstone images to segment the principal minerals and porosity as a proof of concept and then continue to add segmentation of more sedimentary features to the model.

The following are examples of training images divided in 256x256 size.
First is the parallel polarized light (PPL) image, second is the crossed polarized light (XPL) image and last is the annotated mask.
<img src="images/sanity_check_1.jpg" width="800" height="300">
<img src="images/sanity_check_2.jpg" width="810" height="300">
<img src="images/sanity_check_3.jpg" width="810" height="300">

Example of initial test with predictions using U-net with a model trained with very few images.
<img src="images/prediction1.png" width="550" height="400">

