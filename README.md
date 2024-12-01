# Information Gap based Knowledge Distillation #
## Abstract ##
Facial Expression Recognition (FER) with occlusion presents a challenging task in computer vision because facial occlusions result in poor visual data features. Recently, the region attention technique has been introduced to address this problem by researchers, which make the model perceive occluded regions of the face and prioritize the most discriminative non-occluded regions. However, in real-world scenarios, facial images are influenced by various factors, including hair, masks and sunglasses, making it difficult to extract high-quality features from these occluded facial images. This inevitably limits the effectiveness of attention mechanisms. In this paper, we observe a correlation in facial emotion features from the same image, both with and without occlusion. This correlation contributes to addressing the issue of facial occlusions. To this end, we propose a Information Gap based Knowledge Distillation (IGKD) to explore the latent relationship. Specifically, our approach involves feeding non-occluded and masked images into separate teacher and student networks. Due to the incomplete emotion information in the masked images, there exists an information gap between the teacher and student networks. During training, we aim to minimize this gap to enable the student network to learn this relationship. To enhance the teacher's guidance, we introduce a joint learning strategy where the teacher conducts knowledge distillation on the student during the training of the teacher. Additionally, we introduce two novel constraints, called knowledge learn and knowledge feedback loss, to supervise and optimize both the teacher and student networks. The reported experimental results show that IGKD outperforms other algorithms on four benchmark datasets. Specifically, our IGKD achieves 87.57% on Occlusion-RAF-DB, 87.33% on Occlusion-FERPlus, 64.86% on Occlusion-AffectNet, and 73.25% on FED-RO, clearly demonstrating its effectiveness and robustness. 

![Framework](https://github.com/lzh-captain/Information-Gap-based-Knowledge-Distillation/blob/main/images/Framework.png)

## Train
###Environmental requirements:  
    PyTorch == 1.12.1  
    torchvision == 0.13.1
###Dataset
```
RAF-DB/
    Annotated/
        tarin_labels.txt
        val_labels.txt
    train/
        train_images1.jpg
        train_images2.jpg
        ...
    val/
        val_images1.jpg
        val_images2.jpg
        ...	
```
###Pretrained backbone model
Download the pretrained ResNet-18 and ResNet-50 model and then put it under the checkpoint directory.
###Train the model
```
python train.py --teacher-backbone resnet18 --students-backbone resnet18 --data ./data-set/RAF-DB --num-classes 7 --s_checkpoint ./checkpoints/resnet18_msceleb.pth --t_checkpoint ./checkpoints/resnet18_msceleb.pth
```
##Results
![Result](https://github.com/lzh-captain/Information-Gap-based-Knowledge-Distillation/blob/main/images/Result.png)


