# Conditional_GAN


Conditional generative adversarial network (cGAN) on CelebA dataset is implemented to generate synthetic human faces given some conditions. cGAN composed of two key components- generator and discriminator. The generator acts as a sampler to generate images from random gaussian distribution with some conditions. Here, first we train the discriminator to distinguish real vs fake data. In the next step, we train the generator to make fake data which can fool the discriminator. We then continue this discriminator and generator training for multiple epochs. 


Evaluation Metric:

FID Score: 15.69\\
IS Score: 2.12
