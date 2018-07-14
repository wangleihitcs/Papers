# 简介
本人陆续读的一些论文，图像处理相关，侧重于医学报告生成。

## 经典任务
* 检测(detection)
	* DeepRadiologyNet - Radiologist Level Pathology Detection in CT Head Images
	* 肺部CT图像病变区域检测方法
	* 基于定量影像组学的肺肿瘤良恶性预测方法
* 增强(enhace)
	* 超分(super resolution)
		* Image Super-Resolution Using Deep Convolutional Networks
		* Deeply-Recursive Convolutional Network for Image Super-Resolution
* 分割(segmentation)
	* U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015 MICCAI
	* A 3D Coarse-to-Fine Framework for Automatic Pancreas Segmentation

## 生成文字类任务
* 医学报告生成(medical report generation)
	* CNN-RNN
		* (++++)TieNet Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays, CVPR 2018
		* (+++)On the Automatic Generation of Medical Imaging Reports, ACL 2018
		* (+++)Hybrid Retrieval-Generation Reinforced Agent for Medical Image Report Generation

* 图像描述生成(caption)
	* CNN-RNN
		* Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, ICML 2017([code](https://github.com/kelvinxu/arctic-captions))
		* Show and Tell: A Neural Image Caption Generator, CVPR 2015
		* Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge, PAMI 2016([code](https://github.com/tensorflow/models/tree/master/research/im2txt))
		* Areas of Attention for Image Captioning, ICCV 2017
	* CNN-CNN
		* Convolutional Image Captioning, CVPR 2018 

* 图像问答(visual question answering)
	* CNN-RNN
		* Multi-level Attention Networks for Visual Question Answering, CVPR 2017 
