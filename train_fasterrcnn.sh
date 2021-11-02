export CUDA_VISIBLE_DEVICES='2,3'

PRETRAIN_MODEL='/media/apple/faster-rcnn.pytorch/resnet18-5c106cde.pth'
#DATASET_PATH='/home/biometrics/data/coco'
DATASET_PATH='/media/apple/Datasets/coco'
CDC=1 #0.01
RDC=1
EXP_NAME="dummy_adam_Lr0.0001_fasterRCNN_resnet18_binarynet_distillation_head_KLloss_LambdaLR_classAgnostic_CDC${CDC}_RDC${RDC}"
#DATASET_PATH='/home/biometrics/data/VOCdevkit'
LR=0.001
CAPTION='FasterRCNN_binary'
SERVER='Sierra'
#NET='res18'
NET='birealnet18'
python trainval_net.py --o 'adam' --cag --cdc ${CDC} --rdc ${RDC} --scheduler 'LambdaLR' --caption ${CAPTION} --server ${SERVER} --cuda --mGPUs --exp_name $EXP_NAME --lr $LR --dataset='coco' --epochs 12 --net ${NET} #--basenet=$PRETRAIN_MODEL  #--nw 1 --bs 1
#python faster_rcnn/trainval_net.py --dataset='coco' --data_root='/home/biometrics/data/coco' --basenet='/media/Rozhok/BiDet/faster_rcnn/pretrain/resnet18.pth'                                                    