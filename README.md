# DeepCorrect
Devpost: https://devpost.com/software/deep-correct-dl-color-correction-for-color-blindness?ref_content=user-portfolio&ref_feature=in_progress

## Title 
Deep Correct: Deep Learning Color Correction for Color Blindness

## Introduction 
Color vision deﬁciency affects 8% men and one in every 200 women. Our project member, Naveen, is one of them. He has difficulties distinguishing between red and green colored objects. This prompted us to ponder whether deep learning could be used to help with color blindness. After some research, we came across one research paper that uses deep learning to create a color correction program that can aid with object recognition.

We are implementing the following research paper by Petrovic and Fujita: [https://www.researchgate.net/publication/319547415_Deep_Correct_Deep_Learning_Color_Correction_for_Color_Blindness](https://www.researchgate.net/publication/319547415_Deep_Correct_Deep_Learning_Color_Correction_for_Color_Blindness)

Abstract: Color vision deﬁciency affects 8% men and one in every 200 women. There are many different types of color blindness with the red-green as the most common. Most models for color correction are based on physiological models of how people with color vision deﬁciency perceive the world, with the goal of reduc- ing errors derived from the color blindness simulation formula. In this paper we present Deep Correct, a novel Deep Learning based method for color correcting images in order to improve accessibility for people with color vision deﬁciency. The key elements of this work with regard to color blindness are two-fold: 1) we propose a data-driven Deep Learning approach for color correction and 2) we cre- ate an objective, quantitative metric for determining the distinguishability of im- ages. Additionally, as a more general Deep Learning contribution, we propose a new method of training neural networks by utilizing error gradients from pretrained networks in order to train new, smaller networks.

This is an unsupervised learning problem, as the model does not use labels to train. It instead uses an architecture similar to that of a GAN.



## Installs required to run:

Run ```pip install -r requirements.txt``` in the cloned repo

## Poster
![DeepCorrect-Poster](https://user-images.githubusercontent.com/34708350/170667166-95f37a61-5c1f-45b1-b3f4-f6dda24cc467.jpg)
