# Cycle-GAN-for-Depth-Map-Generation
Generative Adversarial Networks have been improved
further to overcome the shortcomings that it initially faced. Cycle is one of the
variations that combines the properties of conditional constraints and cycle consistencies
to effectively use the GAN architecture in Image-to-Image translation tasks. Hence, this
was chosen as the technique to generate the depth images for our project.
Unlike GANs, cycle-GANs can be used to alter a given input image to a distribution of the
target domain. Concretely, an image is taken from input domain D i and then transformed into an image of target domain D t without having a one-to-one mapping between images from the
input to target domain in the training set. Since this architecture has this relaxation of one-
to-one mapping, it becomes quite powerful as the same method could be employed to tackle
variety of problems by varying input and output domain pairs. Since, this method works on
unpaired dataset, it becomes more modular than Pix2Pix architecture. This modularity has been achieved by two step transformations- first by mapping the input image to the target domain and then getting back the original image form
the target domain, Mapping the image to
target domain is done using a generator network and the quality of image is checked by the
discriminator which constantly pushes the generator to perform better.
![alt text](https://github.com/Vineet-Pandey/Cycle-GAN-for-Depth-Map-Generation/blob/master/cyclegan.png)
## Dependencies
1) Pytorch 0.4.1 or newer
2) Python 3.6
3) Matplotlib 3.0
4) OpenCV 3.4 or higher

## Directory Structure
#### Dataset directory

          |--->rgb--->rgb
          |
          |--->depth--->depth
          | 
Dataset-->|
          |
          |--->test_rgb--->rgb
          |
          |--->test_depth--->depth
                 
Corresponding images should go in final directory as shown above.      
#### checkpoints_cyclegan directory
Create a directory named **checkpoints_cyclegan** to save the .pkl files 
#### output directory
Create a directory named **output** to save the inference outputs

