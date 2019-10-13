# 论文阅读笔记：FULLY CONVOLUTIONAL MULTI-CLASS MULTIPLE INSTANCE LEARNING

## Introduce
### Task
Image Semantic Segmentation based on weakly supervised learning.
### Contributions
* Combine MIL(Multiple Instance Learning) with end-to-end FCN(Fully Convolutional Network).
* Propose a multi-class pixel-level loss by a MIL scenario.

## MC MIL Loss
$$(x_{l},y_{l})=\argmax_{\forall(x,y)} \widehat{p}(x,y), \forall l\in L$$
$$MILLoss=-\frac{1}{|L|} \sum\limits_{l \in L}{} \log{\widehat{p}(x,y)}$$
$\widehat{p}(x,y)$ be the FCN output heatmaps($W*H*L$) for the $l^{th}$ class label at location $(x,y)$，$L$ is the class label num.

## Experiments
![experiments](./experiments.png)
Baseline（no classifier）mean without initializing the last layer weight i.e. the classifier layer, Baseline（with classifier）mean with. MIL-FCN mean with initializing the classifier layer and finetune model with MC MIL Loss. Learning rate  is 0.0001,momentum is 0.9, weight decay is 0.0005. Model converges in less than 10000 iterations.

## Details
* If there is no image-level pretraining, the model quickly converges to all background.
* At inference time, the MIL-FCN need a bilinearly interpolations to image resolutions to obtain a pixel wise segmentation.

## Summary
* MIL Loss increase the probability of all class channel heatmap scores.
* MIL Loss is a prior that there are all class labels in a input image even if we don't know the certain location in the image.
