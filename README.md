# SUnCNN
SUnCNN: Sparse Unmixing Using Unsupervised Convolutional Neural Network
SUnCNN  is  the  first  deep  learning-based  technique proposed for sparse unmixing. It uses a deep convolutional encoder-decoder to generate the abundances relying on a spectral library. We reformulate the sparse unmixing into an optimization over the deep network’s parameters. Therefore, the deep network learns  in  an  unsupervised  manner  to  map  a  fixed  input  intothe  sparse  optimum  abundances.  Additionally,  SUnCNN  holds the  sum-to-one  constraint  using  a  softmax  activation  layer.


Note that the model used is a modified version of the DIP software (https://github.com/DmitryUlyanov/deep-image-prior) which is uploaded here and therefore the copyright of it is preseved.

Note that the results reported in the paper are mean values over ten experiments. tol2 variable is the number of runs.

If you use this code please cite the following paper Rasti, B., and Koirala, "SUnCNN: Sparse Unmixing Using Unsupervised Convolutional Neural Network" IEEE Geoscience and Remote Sensing Letters.

DC1: The fractional abundance of endmember 2. From top to bottom SNR of 20, 30, and 40 dB.
![image](https://user-images.githubusercontent.com/61419984/128629548-8d3681f6-5dc3-41fa-90f9-0596e859ca62.png)

