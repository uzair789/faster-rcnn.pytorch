export CUDA_VISIBLE_DEVICES='4,5'

PRETRAIN_MODEL='/media/apple/faster-rcnn.pytorch/resnet18-5c106cde.pth'
#DATASET_PATH='/home/biometrics/data/coco'
DATASET_PATH='/media/apple/Datasets/coco'
EXP_NAME='adam_Lr0.0001_fasterRCNN_resnet18_binarynet_distillation_head_KLloss_LambdaLR'
#DATASET_PATH='/home/biometrics/data/VOCdevkit'
LR=0.001
CAPTION='FasterRCNN_binary'
SERVER='Sierra'
#NET='res18'
NET='birealnet18'
python trainval_net.py --o 'adam' --scheduler 'LambdaLR' --caption ${CAPTION} --server ${SERVER} --cuda --mGPUs --exp_name $EXP_NAME --lr $LR --dataset='coco' --epochs 12 --net ${NET} #--basenet=$PRETRAIN_MODEL  #--nw 1 --bs 1
#python faster_rcnn/trainval_net.py --dataset='coco' --data_root='/home/biometrics/data/coco' --basenet='/media/Rozhok/BiDet/faster_rcnn/pretrain/resnet18.pth'                                                    
