# CMMCSegNet
CMMCSegNet combined with multi-modal image information to overcome the problem of less available samples in medical image segmentation.
The first stage uses Cycle-GAN to realize image translation between modalities, and the second stage uses multi-cascade pix2pix for image segmentation.

## Requirements
Pytorch >=0.8 <br>
torchvision <br>
python>=3.6 <br>

##References
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio." Generative Adversarial Networks."  ArXiv 2014.


J. Zhu, T. Park, P. Isola and A. A. Efros, "Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks," 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2242-2251, doi: 10.1109/ICCV.2017.244.

P. Isola, J. Zhu, T. Zhou and A. A. Efros, "Image-to-Image Translation with Conditional Adversarial Networks," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 5967-5976, doi: 10.1109/CVPR.2017.632.
