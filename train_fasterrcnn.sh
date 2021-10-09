export CUDA_VISIBLE_DEVICES='2,3'

PRETRAIN_MODEL='/media/apple/faster-rcnn.pytorch/resnet18-5c106cde.pth'
#DATASET_PATH='/home/biometrics/data/coco'
DATASET_PATH='/media/apple/Datasets/coco'
EXP_NAME='adam_Lr0.0001_fasterRCNN_resnet18_baseline_full_precision'
#DATASET_PATH='/home/biometrics/data/VOCdevkit'
LR=0.001
CAPTION='FasterRCNN'
SERVER='Sierra'
python trainval_net.py --o 'adam' --caption ${CAPTION} --server ${SERVER} --cuda --mGPUs --exp_name $EXP_NAME --lr $LR --dataset='coco' --epochs 12 --net res18 #--basenet=$PRETRAIN_MODEL  #--nw 1 --bs 1
#python faster_rcnn/trainval_net.py --dataset='coco' --data_root='/home/biometrics/data/coco' --basenet='/media/Rozhok/BiDet/faster_rcnn/pretrain/resnet18.pth'
~                                                    
