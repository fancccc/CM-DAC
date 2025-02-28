# Clinical Prior Guided Cross-Modal Hierarchical Fusion for Accurate Lung Cancer Diagnosis in CT Scans
## Ours framework
![](images/framework.png)
## Abstract
Accurate localization and classification of lung cancer in CT images are crucial for effective clinical treatment. However, existing approaches still face challenges such as redundant information in CT images, ineffective integration of clinical prior knowledge, and the difficulty in distinguishing subtle histological differences across lung cancer subtypes. To address these and enhance classification accuracy, we propose CM-DAC framework.
It adopts a YOLO-based slice detection module to detect and crop lesion areas into fixed sizes, which are then used as input for the Multimodal Contrast Learnning Pretrain (MCLP) module, reducing redundant information interference. 
In MCLP, 3D patches are aligned with clinical records through our Cross-modal Hierarchical Fusion module. This module uses an attention mechanism and residual connections to efficiently integrate image and clinical prior features. To capture subtle histological differences, we employ multi-scale fusion strategies by processing features at different resolutions, enabling the network to pay closer attention to features at all scales.
Simultaneously, a text path employs a medical ontology-driven text augmentation approach to expand category labels into semantic vectors describing morphological features. These vectors are encoded and aligns with fusion feature vectors.
Our CM-DAC outperforms several competitive methods, demonstrating exceptional classification performance.

# Codes
## For trainning
### det model
python ./model/yolo/v11.py
### cls model
python train_cls.py --root /path/your/datasets
## Inference
python train_cls.py --root /path/your/datasets --phase val


