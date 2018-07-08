# 简介
本人陆续读的一些论文，图像处理相关，侧重于医学报告生成。

## 经典任务
* 检测(detection)
	* (+)DeepRadiologyNet - Radiologist Level Pathology Detection in CT Head Images
	* (++)肺部CT图像病变区域检测方法
	* (+)基于定量影像组学的肺肿瘤良恶性预测方法
* 增强(enhace)
	* 超分(super resolution)
		* (+++)Image Super-Resolution Using Deep Convolutional Networks
		* (+++)Deeply-Recursive Convolutional Network for Image Super-Resolution
* 分割(segmentation)
	* (++++)U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015 MICCAI
	* (+)A 3D Coarse-to-Fine Framework for Automatic Pancreas Segmentation

## 生成文字类任务
* 医学报告生成(medical report generation)
	* CNN+LSTN
		* (+++)TieNet Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays, 2018 CVPR
		* (+++)On the Automatic Generation of Medical Imaging Reports, 2018 ACL
		* (++)Hybrid Retrieval-Generation Reinforced Agent for Medical Image Report Generation
* 图像描述生成(caption)
	* Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, 2017 ICML([code](https://github.com/kelvinxu/arctic-captions))
	* Show and Tell: A Neural Image Caption Generator, 2015 CVPR
	* Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge, 2016 PAMI([code](https://github.com/tensorflow/models/tree/master/research/im2txt))
	* Areas of Attention for Image Captioning, 2017 ICCV
