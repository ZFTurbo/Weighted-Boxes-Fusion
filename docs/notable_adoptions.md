## Notable Adoptions and Implementations of WBF methods

- **[WBF-ODAL: Weighted Boxes Fusion for 3D Object Detection from Automotive LiDAR Point Clouds](https://www.researchgate.net/publication/387917705_WBF-ODAL_Weighted_Boxes_Fusion_for_3D_Object_Detection_from_Automotive_LiDAR_Point_Clouds)**
  Dhvani Katkoria, Jaya Sreevalsan-Nair, Mayank Sati, Sunil Karunakaran — *IEEE International Conference on Vehicular Technology and Transportation Systems (ICVTTS)*, 2024.  
  Adapted WBF from 2D images to 3D LiDAR point clouds, creating the "WBF-ODAL" system for fusing 3D bounding-box predictions in autonomous-vehicle object detection.

- **Universal Lymph Node Detection in T2 MRI Using Neural Networks**
  Tejas Sudharshan Mathai, Sungwon Lee, Thomas C. Shen, Zhiyong Lu, Ronald M. Summers — *International Journal of Computer Assisted Radiology and Surgery*, 2023.  
  Applied WBF to merge predictions produced by different neural network models and across training epochs for universal lymph node detection in T2 MRI, using it as the ensemble step that combined model outputs to raise detection accuracy.

- **MoDAR: Using Motion Forecasting for 3D Object Detection in Point Cloud Sequences**
  Yingwei Li, Charles R. Qi, Yin Zhou, Chenxi Liu, Dragomir Anguelov — *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.  
  Used WBF inside the MoDAR system to fuse ten temporal 3D bounding-box predictions from past and future LiDAR frames, suppressing noise and redundancy across temporal predictions for more precise object localization over time.

- **Introducing Multiagent Systems to AV Visual Perception Sub-tasks**
  Alaa Daoud, Corentin Bunel, Maxime Guériau — *13th International Workshop on Agents in Traffic and Transportation (ATT 2024)*.  
  Built a multiagent system creating a decentralized "Agentified Weighted Boxes Fusion" (AWBF) variant for visual perception.

- **Nodule Detection and Generation on Chest X-Rays: NODE21 Challenge**
  Ecem Sogancioglu, Bram van Ginneken, et al. — *IEEE Transactions on Medical Imaging*, 2024.  
  Used WBF to combine the outputs of four distinct, high-performing lung-nodule detection models, with the resulting ensemble matching or exceeding radiologist performance.

- **An Ensemble of Deep Learning Object Detection Models for Anatomical and Pathological Regions in Brain MRI**
  Ramazan Terzi — *Diagnostics*, 2023.  
  Implemented WBF to construct ensembles across nine deep-learning object detection models for anatomical and pathological localization in brain MRI, achieving improved detection performance.

- **Ensemble Fusion for Small Object Detection**
  Hao-Yu Hou, Mu-Yi Shen, Chia-Chi Hsu, En-Ming Huang, Yu-Chen Huang, Yu-Cheng Xia, Chien-Yao Wang, Chun-Yi Lee — *18th International Conference on Machine Vision Applications*, 2023.  
  Applied WBF to combine predictions from several detector variants for small-object detection (birds), identifying WBF as the most effective ensembling strategy tested for their bird-spotting challenge.

- **Seamless Iterative Semi-supervised Correction of Imperfect Labels in Microscopy Images**
  Marawan Elbatel, Christina Bornberg, Manasi Kattel, Enrique Almar, Claudio Marrocco, Alessandro Bria — *DART 2022 (MICCAI Workshop, LNCS 13542)*.  
  Implemented WBF at the pseudo-label generation step, combining it with test-time augmentation to merge overlapping bounding boxes into a cleaner, more confident set of labels.

- **Transformer-based Mass Detection in Digital Mammograms**
  Amparo S. Betancourt Tarifa, Claudio Marrocco, Mario Molinara, Francesco Tortorella, Alessandro Bria — *Journal of Ambient Intelligence and Humanized Computing*, 2023.  
  Used WBF to merge predictions from architecturally different mass-detection models (convolutional and Swin-transformer based) in digital mammograms, producing their best-performing ensemble system.
