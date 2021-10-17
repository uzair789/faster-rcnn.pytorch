
#ROOT='/media/Rozhok/BiDet/logs/coco/bidet18_IB/models/'

#CHECKPOINT='/media/Rozhok/BiDet/logs/coco/bidet18_IB/models/model_29_loss_0.8189_lr_1.0000000000000002e-06_rpn_cls_0.1711_rpn_bbox_0.1005_rcnn_cls_0.3346_rcnn_bbox_0.2113_rpn_prior_0.0_rpn_reg_0.0014_head_prior_0.0_head_reg_0.0.pth'


export CUDA_VISIBLE_DEVICES='0'
EXP_ID='DIS-513'
NET='birealnet18'
python test_net.py --cuda --dataset 'coco' --net ${NET} --exp_id ${EXP_ID}
