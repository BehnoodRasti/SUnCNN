# SUnCNN
SUnCNN: Sparse Unmixing Using Unsupervised Convolutional Neural Network
SUnCNN  is  the  first  deep  learning-based  technique proposed for sparse unmixing. It uses a deep convolutional encoder-decoder to generate the abundances relying on a spectral library. We reformulate the sparse unmixing into an optimization over the deep network’s parameters. Therefore, the deep network learns  in  an  unsupervised  manner  to  map  a  fixed  input  intothe  sparse  optimum  abundances.  Additionally,  SUnCNN  holds the  sum-to-one  constraint  using  a  softmax  activation  layer.


Not that the model used is a modified version of the DIP software (https://github.com/DmitryUlyanov/deep-image-prior) which is uploaded here and therefore the copyright of it is preseved.

If you use this code please cite the following paper Rasti, B., and Koirala, "SUnCNN: Sparse Unmixing Using Unsupervised Convolutional Neural Network"
