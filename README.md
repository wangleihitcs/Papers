# Intro
Combine CV with NLP tasks，focus on Medical Report Generation、Image/Video Captioning、VQA、Scene Text Detection、Anchor-free Object Detection.
- [Image/Video Captioning](#Image-Video-Captioning)
- [Paragraph Description Generation](#Paragraph-Description-Generation)
- [Visual Question Answering](#Visual-Question-Answering)
- [Medical Report Generation](#Medical-Report-Generation)
- [Medical Image Processing](#Medical-Image-Processing)
- [Natural Image Tasks](#Natural-Image-Tasks)
- [Scene Text Detection and Recognition](#Scene-Text-Detection-and-Recognition)
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
- CNN-RNN
	* Learning to Read Chest X-Rays- Recurrent Neural Cascade Model for Automated Image Annotation, CVPR 2016[(pdf)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7780643)
	* **TieNet Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays**, Xiaosong Wang et at, CVPR 2018, NIH[(pdf)](http://www.cs.jhu.edu/~lelu/publication/TieNet_CVPR2018_spotlight.pdf)[(author's homepage)](https://xiaosongwang.github.io)
	* **On the Automatic Generation of Medical Imaging Reports**, Baoyu Jing et al, ACL 2018, CMU[(pdf)](https://arxiv.org/pdf/1711.08195.pdf)[(author's homepage)](http://www.cs.cmu.edu/~pengtaox/)
	* **Multimodal Recurrent Model with Attention for Automated Radiology Report Generation**, Yuan Xue, MICCAI 2018, PSU[(pdf)](https://faculty.ist.psu.edu/suh972/Xue-MICCAI2018.pdf)
	* Attention-Based Abnormal-Aware Fusion Network for Radiology Report Generation, Xiancheng Xie et al, 2019, Fudan University
	* Addressing Data Bias Problems for Chest X-ray Image Report Generation, Philipp Harzig et al, 2019, University of Augsburg Augsburg[(pdf)](https://arxiv.org/pdf/1908.02123.pdf)

- Reinforcement Learning
	* **Hybrid Retrieval-Generation Reinforced Agent for Medical Image Report Generation**, Christy Y. Li et al, NIPS 2018, CMU[(pdf)](https://arxiv.org/pdf/1805.08298.pdf)[(author's homepage)](https://www.cs.cmu.edu/~zhitingh/)

- Knowledge Graph
	* **Knowledge-Driven Encode, Retrieve, Paraphrase for Medical Image Report Generation**, Christy Y. Li et al, AAAI 2019, DU[(pdf)](https://www.aaai.org/Papers/AAAI/2019/AAAI-LiChristy.629.pdf)
		
- Other
	* TextRay Mining Clinical Reports to Gain a Broad Understanding of Chest X-rays, 2018 MICCAI[(pdf)](https://arxiv.org/abs/1806.02121)

- Blogs
	* [医学报告生成综述](https://blog.csdn.net/wl1710582732/article/details/85345285)


### Medical Image Processing
#### Common Datasets
- NIH Chest X-ray8/14[(download link)](https://nihcc.app.box.com/v/ChestXray-NIHCC)[(kaggle's download link)](https://www.kaggle.com/nih-chest-xrays/data)
	* ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, CVPR 2017, NIH[(pdf)](https://arxiv.org/pdf/1705.02315.pdf)

- Open-i Chest X-Ray[(download link)](https://openi.nlm.nih.gov/faq.php?it=xg)

- Radiology Objects in COntext(ROCO)
	* Radiology Objects in COntext (ROCO): A Multimodal Image Dataset, MICCAI 2018[(intro)](https://labels.tue-image.nl/wp-content/uploads/2018/09/AM-04-ROCO_Labels_MICCAI_2018.pdf)[(pdf)](https://labels.tue-image.nl/wp-content/uploads/2018/09/AM-04.pdf)[(download)](https://github.com/razorx89/roco-dataset)

#### Medical Tasks
- Detection
	* CheXNet- Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning, 2018 吴恩达
	* **Attention-Guided Curriculum Learning for Weakly Supervised Classification and Localization of Thoracic Diseases on Chest Radiographs**, Yuxing Tang et at, MICCAI-MLMI oral 2018, NIH[(pdf)](https://arxiv.org/pdf/1807.07532.pdf)
	* DeepRadiologyNet - Radiologist Level Pathology Detection in CT Head Images
	* 肺部CT图像病变区域检测方法
	* 基于定量影像组学的肺肿瘤良恶性预测方法

- Enhance
	* Super Resolution
		* Image Super-Resolution Using Deep Convolutional Networks
		* Deeply-Recursive Convolutional Network for Image Super-Resolution
- Segmentation
	* U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015 MICCAI
	* A 3D Coarse-to-Fine Framework for Automatic Pancreas Segmentation

### Natural Image Tasks
#### Classification
* VGG: Very Deep Convolutional NetWorks for Large-Scale Image Recognition, Karen Simonyan et at, ICLR 2015[(pdf)](https://arxiv.org/pdf/1409.1556.pdf)
* Inception：Going Deeper with Convolutions, Christian Szegedy et al, CVPR 2015, Google[(pdf)](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
* ResNet：Deep Residual Learning for Image Recognition, Kaiming He et al, CVPR 2016, Microsoft Research[(pdf)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)[(code)](https://github.com/KaimingHe/deep-residual-networks)[(blog)](https://blog.csdn.net/wspba/article/details/56019373)
* SENet：Squeeze-and-Excitation Networks, Jie Hu et al, CVPR 2018,  Momenta(中国无人驾驶公司，曹旭东创立) and Oxford University[(pdf)](https://www.robots.ox.ac.uk/~vgg/publications/2018/Hu18/hu18.pdf)[(code)](https://github.com/hujie-frank/SENet)[(blog)](https://blog.csdn.net/wangkun1340378/article/details/79092001)

#### Detection
- Weakly-supervised
	* Learning Deep Features for Discriminative Localization, Bolei Zhou et al, CVPR 2016, MIT[(pdf)](https://arxiv.org/pdf/1512.04150.pdf)[(code)](https://github.com/metalbubble/CAM)[(note)](./netural%20image%20tasks/detection/Learning%20Deep%20Features%20for%20Discriminative%20Localization/note.md)

- Anchor-based
	* YOLO9000- Better, Faster, Stronger, Joseph Redmon et al, CVPR 2017[(pdf)](http://web.eng.tau.ac.il/deep_learn/wp-content/uploads/2018/01/YOLO9000.pdf)[(project)](https://pjreddie.com/darknet/yolo/)[(code)](https://github.com/longcw/yolo2-pytorch)
	* **SSD: Single Shot MultiBox Detector**, Wei Liu et al, ECCV 2016, UNC Chapel Hill[(pdf)](https://www.cs.unc.edu/~wliu/papers/ssd.pdf)[(code)](https://github.com/weiliu89/caffe/tree/ssd)[(blog)](https://blog.csdn.net/u010167269/article/details/52563573)

- Anchor-free
	* YOLO, **You Only Look Once- Unified, Real-Time Object Detection**, Joseph Redmon et al, CVPR 2016[(pdf)](https://pjreddie.com/media/files/papers/yolo.pdf)[(note)](./netural%20image%20tasks/detection/You%20Only%20Look%20Once-%20Unified%20Real-Time%20Object%20Detection/note.md)
	* CornerNet, **CornerNet: Detecting Objects as Paired Keypoints**, Hei Law et al, ECCV 2018, Michigan University[(pdf)](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.pdf)[(code)](https://github.com/princeton-vl/CornerNet)[(blog)](http://www.deepsmart.ai/508.html)
	* FCOS, FCOS: Fully Convolutional One-Stage Object Detection, Zhi Tian et al, ICCV 2019, Adelaide University[(pdf)](https://arxiv.org/pdf/1904.01355.pdf)[(code)](https://github.com/tianzhi0549/FCOS/)[(blog)](https://blog.csdn.net/qiu931110/article/details/89073244)
	* CenterNet, **Objects as Points**, Xingyi Zhou et al, 2019, UT Austin[(pdf)](https://arxiv.org/pdf/1904.07850.pdf)[(code)](https://github.com/xingyizhou/CenterNet)

- Others
	* Bag of Freebies for Training Object Detection Neural Networks, Zhi Zhang et al, 2019, Amazon 李沐[(pdf)](https://arxiv.org/pdf/1902.04103.pdf)
	* **Deformable Convolutional Networks**, Jifeng Dai et al, ICCV 2017, Microsoft Research Asia[(pdf)](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf)[(code)](https://github.com/msracver/Deformable-ConvNets)

#### Segmentation
* **Mask R-CNN**, Kaiming He et al, ICCV 2017(Best Paper), Facebook AI Research (FAIR)[(pdf)](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)[(code)](https://github.com/matterport/Mask_RCNN)


### Scene Text Detection and Recognition
#### Overview
- **Scene Text Detection and Recognition: The Deep Learning Era**, Shangbang Long et al, 2018, Megvii[(pdf)](https://arxiv.org/pdf/1811.04256.pdf)[(releated sources)](https://github.com/Jyouhou/SceneTextPapers)

#### Scene Text Detection
- Pipelines
	- Multi Step
		* Multi-Oriented Text Detection with Fully Convolutional Networks, Zheng Zhang et al, CVPR 2016, HUST[(pdf)](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Multi-Oriented_Text_Detection_CVPR_2016_paper.pdf)[(code)](https://github.com/stupidZZ/FCN_Text)[(blog)](https://www.cnblogs.com/lillylin/p/6102708.html)
	- Simplified Pipeline
		- Anchor Based
			* TextBoxes, **TextBoxes: A Fast Text Detector with a Single Deep Neural Network**, Minghui Liao et al, AAAI 2017, HUST 白翔组[(pdf)](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14202/14295)[(code)](https://github.com/MhLiao/TextBoxes)[(blog)](https://www.cnblogs.com/lillylin/p/6204099.html)
			* EAST, **EAST: An Efficient and Accurate Scene Text Detector**, Xinyu Zhou et al, CVPR 2017, Megvii 姚聪组[(pdf)](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf)[(code)](https://github.com/argman/EAST)[(blog)](https://blog.csdn.net/sparkexpert/article/details/77987654)
		- Region Proposal
			* FEN, Feature Enhancement Network: A Refined Scene Text Detector, Sheng Zhang et al, AAAI 2018, South China University of Technology 金连文组[(pdf)](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16806/15981)
- Predict Units
	- Text-Line Level
		* TextBoxes、EAST、FEN
	- Sub-Text-Line Level
		- Pixel Level
			* SSTD, **Single Shot Text Detector with Regional Attention**, Pan He et al, ICCV 2017, Oxford 黄伟林组[(pdf)](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Single_Shot_Text_ICCV_2017_paper.pdf)[(code)](https://github.com/BestSonny/SSTD)
		- Components Level(Word or Character Level)
			* SegLink, **Detecting Oriented Text in Natural Images by Linking Segments**, Baoguang Shi et al, CVPR 2017, HUST 白翔组[(pdf)](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shi_Detecting_Oriented_Text_CVPR_2017_paper.pdf)[(code)](https://github.com/bgshih/seglink)[(blog)](https://zhuanlan.zhihu.com/p/37781277)
- Target
	- Long Text
		* SegLink
	- Irregular Shapes
		* TextSnake, TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes, Shangbang Long et al, ECCV 2018, PU 姚聪组[(pdf)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Shangbang_Long_TextSnake_A_Flexible_ECCV_2018_paper.pdf)

#### Scene Text Recognition
- Connectionist Temporal Classification
	* CRNN, **An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition**, Baoguang Shi et al, TPAMI 2017, HUST 白翔组[(pdf)](https://arxiv.org/pdf/1507.05717.pdf)[(code)](https://github.com/bgshih/crnn)
- Attention Mechanism
	* R2AM, Recursive Recurrent Nets with Attention Modeling for OCR in the Wild, Chen-Yu Lee et al, CVPR 2016, UC San Diego[(pdf)](http://openaccess.thecvf.com/content_cvpr_2016/papers/Lee_Recursive_Recurrent_Nets_CVPR_2016_paper.pdf)
	* Robust Scene Text Recognition with Automatic Rectification, Baoguang Shi et al, CVPR 2016, HUST 白翔组[(pdf)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Robust_Scene_Text_CVPR_2016_paper.pdf)

#### End-to-End Scene Text Detection and Recognition
* Deep TextSpotter, **Deep TextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework**, Michal Busta et al, ICCV 2017, Czech Technical University[(pdf)](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busta_Deep_TextSpotter_An_ICCV_2017_paper.pdf)[(code)](https://github.com/MichalBusta/DeepTextSpotter)
* See, **SEE: Towards Semi-Supervised End-to-End Scene Text Recognition**, Christian Bartz et al, AAAI 2018, Potsdam[(pdf)](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16270/16248)[(code)](https://github.com/Bartzi/see)
* Mask TextSpotter, Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes, Pengyuan Lyu et al, HUST 白翔组[(pdf)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Pengyuan_Lyu_Mask_TextSpotter_An_ECCV_2018_paper.pdf)

### Metrics
- BLEU
	* **BLEU: a method for automatic evaluation of machine translation**, Kishore Papineni et al, ACL 2002[(pdf)](https://www.aclweb.org/anthology/P02-1040.pdf)
- CIDEr
	* CIDEr: Consensus-based Image Description Evaluation, CVPR 2015[(pdf)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)[(note)](https://blog.csdn.net/wl1710582732/article/details/84202254)

### Others
- Visual Commonsense Reasoning(VCR-视觉常识推理)
	* **From Recognition to Cognition- Visual Commonsense Reasoning**, Rowan Zeller et al, 2018, Paul G. Allen School[(homepage)](http://visualcommonsense.com)[(pdf)](https://arxiv.org/pdf/1811.10830.pdf)
- Language Model(语言模型)
	* Transformer：**Attention Is All You Need**, Ashish Vaswani et al, NIPS 2017, Google Brain/Research[(pdf)](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)[(code)](https://github.com/jadore801120/attention-is-all-you-need-pytorch)[(blog)](https://medium.com/@cyeninesky3/attention-is-all-you-need-%E5%9F%BA%E6%96%BC%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%A9%9F%E5%88%B6%E7%9A%84%E6%A9%9F%E5%99%A8%E7%BF%BB%E8%AD%AF%E6%A8%A1%E5%9E%8B-dcc12d251449)
	* **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, Jacob Devlin et al, 2018, Googel AI Language[(pdf)](https://arxiv.org/pdf/1810.04805.pdf)[(code)](https://github.com/google-research/bert)[(slides)](https://nlp.stanford.edu/seminar/details/jdevlin.pdf)
	* ELMo：**Deep contextualized word representations**, Matthew E. Peters et al, NAACL 2018, Paul G. Allen School[(homepage)](https://allennlp.org/elmo)[(pdf)](https://arxiv.org/pdf/1802.05365.pdf)[(code-tf)](https://github.com/allenai/bilm-tf)

	
- Teacher Forcing Policy
	* A learning algorithm for continually running fully recurrent neural networks, Ronald et al, Neural Computation 1989[(pdf)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.9724&rep=rep1&type=pdf)[(node)](https://blog.csdn.net/wl1710582732/article/details/88636852)
	* Professor Forcing: A New Algorithm for Training Recurrent Networks, Alex Lamb et al, NIPS 2016[(pdf)](https://arxiv.org/pdf/1610.09038.pdf)
