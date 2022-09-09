# Conditional_GAN


Conditional generative adversarial network (cGAN) on CelebA dataset is implemented to generate synthetic human faces given some conditions. cGAN composed of two key components- generator and discriminator. The generator acts as a sampler to generate images from random gaussian distribution with some conditions. Here, first we train the discriminator to distinguish real vs fake data. In the next step, we train the generator to make fake data which can fool the discriminator. We then continue this discriminator and generator training for multiple epochs. 

# Dataset Description

In this project, we have utilized the CelebA dataset for implementing cGAN. The dataset has 202, 559 celebrity images, each with 40 attributes. Although the original image was $218 \times 178 \times 3$, for this project the cGAN will be implemented into smaller images of size $64 \times 64 \times 3$. Also, out of 40 attributes, this projects uses only 5 of them. These attricutes are: [Black hair, Male, Oval Face, Smiling, and Young]. A conditional vector fulfilling all these conditions will have vector elements as [1,1,1,1,1]. If anyone of the conditions are missing, then the vector element of these particular position will be replaced by 0. AN outlook of the CelebA dataset is shown in the following below. 


## Evaluation Metric:

* FID Score: 15.69
* IS Score: 2.12

## Dataset: 

https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
