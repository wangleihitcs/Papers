# Intro
Combine CV with NLP tasks，place emphasis on Image/Video Captioning、VQA、Paragraph Description Generation and Medical Report Generation.
- [Image/Video Captioning](#Captioning)
- [Paragraph Description Generation](#Paragraph-Description-Generation)
- [Visual Question Answering](#Visual-Question-Answering)
- [Medical Report Generation](#Medical-Report-Generation)
- [Medical Image Processing](#Medical-Image-Processing)
- [Medical Datasets](#Medical-Datasets)
- [Natural Image Tasks](#Natural-Image-Tasks)

## Papers and Codes/Notes
### <div id="Captioning">Image/Video Captioning</div>
- CNN-RNN
	* Learning to Read Chest X-Rays- Recurrent Neural Cascade Model for Automated Image Annotation, CVPR 2016
	* TieNet Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays, CVPR 2018[(author's homepage)](https://xiaosongwang.github.io)
	* On the Automatic Generation of Medical Imaging Reports, ACL 2018, CMU[(author's homepage)](http://www.cs.cmu.edu/~pengtaox/)
	* Multimodal Recurrent Model with Attention for Automated Radiology Report Generation, MICCAI 2018, PSU
	* End-to-End Video Captioning with Multitask Reinforcement Learning
	* Move Forward and Tell- A Progressive Generator of Video Descriptions, ECCV 2018, CUHK

- CNN-CNN
	* Convolutional Image Captioning, CVPR 2018([code](https://github.com/aditya12agd5/convcap))

- Reinforcement Learning
	* Improving Reinforcement Learning Based Image Captioning with Natural Language Prior, 2018 

- Others
	* A Neural Compositional Paradigm for Image Captioning, NIPS 2018, CUHK

### Paragraph Description Generation
- CNN-RNN
	* DenseCap: Fully Convolutional Localization Networks for Dense Captioning, CVPR 2016, Standford[(homepage)](https://cs.stanford.edu/people/karpathy/densecap/)[(code)](https://github.com/jcjohnson/densecap)
	* A Hierarchical Approach for Generating Descriptive Image Paragraphs, CVPR 2017[(homepage)](https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html)[(dense-caption code)](https://github.com/InnerPeace-Wu/densecap-tensorflow)
	* Recurrent Topic-Transition GAN for Visual Paragraph Generation, ICCV 2017
	* Diverse and Coherent Paragraph Generation from Images, ECCV 2018[(code)](https://github.com/metro-smiles/CapG_RevG_Code)

### Visual Question Answering
* CNN-RNN
	* Multi-level Attention Networks for Visual Question Answering, CVPR 2017 
	* Motion-Appearance Co-Memory Networks for Video Question Answering, 2018
	* Deep Attention Neural Tensor Network for Visual Question Answering, ECCV 2018, HIT

### Medical Report Generation
* CNN-RNN
	* Learning to Read Chest X-Rays- Recurrent Neural Cascade Model for Automated Image Annotation, CVPR 2016
	* TieNet Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays, CVPR 2018[(author's homepage)](https://xiaosongwang.github.io)
	* On the Automatic Generation of Medical Imaging Reports, ACL 2018, CMU[(author's homepage)](http://www.cs.cmu.edu/~pengtaox/)
	* Multimodal Recurrent Model with Attention for Automated Radiology Report Generation, MICCAI 2018, PSU

- Reinforcement Learning
	* Hybrid Retrieval-Generation Reinforced Agent for Medical Image Report Generation, NIPS 2018, CMU[(author's homepage)](https://www.cs.cmu.edu/~zhitingh/)
		
- Other
	* TextRay Mining Clinical Reports to Gain a Broad Understanding of Chest X-rays, 2018

### Medical Image Processing
- 分类(classification)
	* CheXNet- Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning, 2018 吴恩达
	
- 检测(detection)
	* DeepRadiologyNet - Radiologist Level Pathology Detection in CT Head Images
	* 肺部CT图像病变区域检测方法
	* 基于定量影像组学的肺肿瘤良恶性预测方法

- 增强(enhace)
	* 超分(super resolution)
		* Image Super-Resolution Using Deep Convolutional Networks
		* Deeply-Recursive Convolutional Network for Image Super-Resolution
- 分割(segmentation)
	* U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015 MICCAI
	* A 3D Coarse-to-Fine Framework for Automatic Pancreas Segmentation

### Medical Datasets
- Chest X-Ray
	* NIH Chest X-Ray([download link])(https://nihcc.app.box.com/v/ChestXray-NIHCC)
		* ChestX-ray8 Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, CVPR 2017, NIH

### Natural Image Tasks
- Detection
	* You Only Look Once- Unified, Real-Time Object Detection, CVPR 2016

