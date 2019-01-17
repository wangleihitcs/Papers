# Intro
Combine CV with NLP tasks，place emphasis on Image/Video Captioning、VQA、Paragraph Description Generation and Medical Report Generation.
- [Image/Video Captioning](#Image-Video-Captioning )
- [Paragraph Description Generation](#Paragraph-Description-Generation)
- [Visual Question Answering](#Visual-Question-Answering)
- [Medical Report Generation](#Medical-Report-Generation)
- [Medical Image Processing](#Medical-Image-Processing)
- [Medical Datasets](#Medical-Datasets)
- [Natural Image Tasks](#Natural-Image-Tasks)
- [Metrics](#Metrics)
- [Others](#Others)

## Papers and Codes/Notes
### Image Video Captioning 
- CNN-RNN
	* **Show and Tell: A Neural Image Caption Generator**, Oriol Vinyals et al, CVPR 2015, Google[(pdf)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)
	* **Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**, Kelvin Xu et at, ICML 2015[(pdf)](https://arxiv.org/pdf/1502.03044.pdf)[(code)](https://github.com/kelvinxu/arctic-captions)
	* Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge, PAMI 2016[(pdf)](https://arxiv.org/pdf/1609.06647.pdf)[(code)](https://github.com/tensorflow/models/tree/master/research/im2txt)
	* Areas of Attention for Image Captioning, ICCV 2017[(pdf)](https://arxiv.org/pdf/1612.01033.pdf)
	* Rethinking the Form of Latent States in Image Captioning, ECCV 2018, CUHK[(pdf)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bo_Dai_Rethinking_the_Form_ECCV_2018_paper.pdf)
	* Recurrent Fusion Network for Image Captioning, ECCV 2018, Tencent AI Lab, 复旦[(pdf)](https://arxiv.org/pdf/1807.09986.pdf)
	* Move Forward and Tell- A Progressive Generator of Video Descriptions, ECCV 2018, CUHK[(pdf)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yilei_Xiong_Move_Forward_and_ECCV_2018_paper.pdf)
	* Video Paragraph Captioning Using Hierarchical Recurrent Neural Networks, CVPR 2016[(pdf)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yu_Video_Paragraph_Captioning_CVPR_2016_paper.pdf)

- CNN-CNN
	* Convolutional Image Captioning, CVPR 2018[(pdf)](https://arxiv.org/pdf/1711.09151.pdf)[(code)](https://github.com/aditya12agd5/convcap)

- Reinforcement Learning
	* Improving Reinforcement Learning Based Image Captioning with Natural Language Prior, 2018, Tencent/IBM[(pdf)](https://arxiv.org/pdf/1809.06227.pdf)
	* End-to-End Video Captioning with Multitask Reinforcement Learning[(pdf)](https://arxiv.org/pdf/1803.07950.pdf)

- Others
	* A Neural Compositional Paradigm for Image Captioning, NIPS 2018, CUHK[(pdf)](https://arxiv.org/pdf/1810.09630.pdf)

### Paragraph Description Generation
- CNN-RNN
	* **DenseCap: Fully Convolutional Localization Networks for Dense Captioning**, Justin Johnson et al, CVPR 2016, Standford[(homepage)](https://cs.stanford.edu/people/karpathy/densecap/)[(code)](https://github.com/jcjohnson/densecap)
	* **A Hierarchical Approach for Generating Descriptive Image Paragraphs**, Jonathan Krause et al, CVPR 2017, Stanford[(homepage)](https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html)[(dense-caption code)](https://github.com/InnerPeace-Wu/densecap-tensorflow)
	* Recurrent Topic-Transition GAN for Visual Paragraph Generation, ICCV 2017
	* Diverse and Coherent Paragraph Generation from Images, ECCV 2018[(code)](https://github.com/metro-smiles/CapG_RevG_Code)

### Visual Question Answering
- CNN-RNN
	* Multi-level Attention Networks for Visual Question Answering, CVPR 2017 
	* Motion-Appearance Co-Memory Networks for Video Question Answering, 2018
	* Deep Attention Neural Tensor Network for Visual Question Answering, ECCV 2018, HIT
	* **Question-Guided Hybrid Convolution for Visual Question Answering**, Peng Gao et al, ECCV 2018, CUHK[(pdf)](https://arxiv.org/pdf/1808.02632.pdf)

### Medical Report Generation
- [Notes](https://blog.csdn.net/wl1710582732/article/details/85345285)
- CNN-RNN
	* Learning to Read Chest X-Rays- Recurrent Neural Cascade Model for Automated Image Annotation, CVPR 2016[(pdf)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7780643)
	* **TieNet Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays**, Xiaosong Wang et at, CVPR 2018, NIH[(pdf)](http://www.cs.jhu.edu/~lelu/publication/TieNet_CVPR2018_spotlight.pdf)[(author's homepage)](https://xiaosongwang.github.io)
	* **On the Automatic Generation of Medical Imaging Reports**, Baoyu Jing et al, ACL 2018, CMU[(pdf)](https://arxiv.org/pdf/1711.08195.pdf)[(author's homepage)](http://www.cs.cmu.edu/~pengtaox/)
	* **Multimodal Recurrent Model with Attention for Automated Radiology Report Generation**, Yuan Xue, MICCAI 2018, PSU

- Reinforcement Learning
	* **Hybrid Retrieval-Generation Reinforced Agent for Medical Image Report Generation**, Christy Y. Li et al, NIPS 2018, CMU[(pdf)](https://arxiv.org/pdf/1805.08298.pdf)[(author's homepage)](https://www.cs.cmu.edu/~zhitingh/)
		
- Other
	* TextRay Mining Clinical Reports to Gain a Broad Understanding of Chest X-rays, 2018 MICCAI[(pdf)](https://arxiv.org/abs/1806.02121)

### Medical Image Processing
- 检测(detection)
	* CheXNet- Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning, 2018 吴恩达
	* **Attention-Guided Curriculum Learning for Weakly Supervised Classification and Localization of Thoracic Diseases on Chest Radiographs**, Yuxing Tang et at, MICCAI-MLMI oral 2018, NIH[(pdf)](https://arxiv.org/pdf/1807.07532.pdf)
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
- NIH Chest X-ray8/14[(download link)](https://nihcc.app.box.com/v/ChestXray-NIHCC)[(kaggle's download link)](https://www.kaggle.com/nih-chest-xrays/data)
	* ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, CVPR 2017, NIH[(pdf)](https://arxiv.org/pdf/1705.02315.pdf)
- Open-i Chest X-Ray[(download link)](https://openi.nlm.nih.gov/faq.php?it=xg)
- Radiology Objects in COntext(ROCO)
	* Radiology Objects in COntext (ROCO): A Multimodal Image Dataset, MICCAI 2018[(intro)](https://labels.tue-image.nl/wp-content/uploads/2018/09/AM-04-ROCO_Labels_MICCAI_2018.pdf)[(pdf)](https://labels.tue-image.nl/wp-content/uploads/2018/09/AM-04.pdf)[(download)](https://github.com/razorx89/roco-dataset)

### Natural Image Tasks
- Detection
	* **You Only Look Once- Unified, Real-Time Object Detection**, Joseph Redmon et al, CVPR 2016[(pdf)](https://pjreddie.com/media/files/papers/yolo.pdf)
	* YOLO9000- Better, Faster, Stronger, Joseph Redmon et al, CVPR 2017[(pdf)](http://web.eng.tau.ac.il/deep_learn/wp-content/uploads/2018/01/YOLO9000.pdf)[(project)](https://pjreddie.com/darknet/yolo/)[(code)](https://github.com/longcw/yolo2-pytorch)

### Metrics
- BLEU
	* **BLEU: a method for automatic evaluation of machine translation**, Kishore Papineni et al, ACL 2002[(pdf)](https://www.aclweb.org/anthology/P02-1040.pdf)
- CIDEr
	* CIDEr: Consensus-based Image Description Evaluation, CVPR 2015[(pdf)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)

### Others
- Visual Commonsense Reasoning(VCR-视觉常识推理)
	* **From Recognition to Cognition- Visual Commonsense Reasoning**, Rowan Zeller et al, 2018, Paul G. Allen School[(homepage)](http://visualcommonsense.com)[(pdf)](https://arxiv.org/pdf/1811.10830.pdf)
- Language Model(语言模型)
	* **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, Jacob Devlin et al, 2018, Googel AI Language[(pdf)](https://arxiv.org/pdf/1810.04805.pdf)[(code)](https://github.com/google-research/bert)
- Word Representations
	* **Deep contextualized word representations**, Matthew E. Peters et al, NAACL 2018, Paul G. Allen School[(homepage)](https://allennlp.org/elmo)[(pdf)](https://arxiv.org/pdf/1802.05365.pdf)[(code-tf)](https://github.com/allenai/bilm-tf)
