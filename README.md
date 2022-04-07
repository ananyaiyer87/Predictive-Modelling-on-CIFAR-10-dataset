# Project: Weakly supervised learning-- label noise and correction


### [Full Project Description](doc/project3_desc.md)

Term: Spring 2022

+ Team # 3
+ Team members
	+ Gexin Chen
	+ Daoyang E
	+ Chang Lu
	+ Shriya Nallamaddi
	+ Ananya Iyer

+ Project summary: In this project, we created a predictive model to conduct image classification. We create a label correction network to clean noisy labels and a convolutional neural network as an image classifier.
	
**Contribution statement**: All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement.  

At first, all of group members met regularly and brainstormed the best models to deal with both label correction and image classification models. All of us have done some searches and collected important and helpful papers to inspire our model design process.  

Chen, Gexin and Lu, Chang collaborated to design and implement the final label correction network on model 2. They were inspired by the paper "Learning From Noisy Large-Scale Datasets With Minimal Supervision" (Andreas Veit et al, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 839-847) and reproduced the algorithm proposed by the paper. Additionally, Gexin and Chang were responsible to ensure that the code of evaluation function and csv output function works well. Also, Gexin communicated with all group members to ensure a smooth working flow on Github through all group members.

Shriya Nallamaddi, Ananya Iyer and Daoyang E worked on implementing the code for Model 1 and 2. They used VGG16, a Convolution Neural Network transfer learning technique to clean the noisy labels. They worked on cleaning and updating the labels, data augmentation, and fine-tuning the model by choosing appropriate parameters to get higher model accuracy and optimal run time.


Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
