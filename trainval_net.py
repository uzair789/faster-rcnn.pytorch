# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.birealnet import birealnet

from icecream import ic
import neptune
from torch.nn import functional as F

##


import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
##





neptune.init('uzair789/Distillation')

class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data



def KL_loss(teacher_output, student_output, id_=1):    
      """Process the KL divergence between the teacher and the student outputs after                 
      the sigmoid operation. The logit maps have been normalized for unit norm.    
                                                                                            
      Arguments:                                                                                     
          teacher_output (torch.Tensor) - Nx(h*w*num_anchors)x80 for class    
                                        - Nx(h*w*num_anchrs)x4 for reg        
                                        teacher not softmaxed                                        
          student_output (torch.Tensor) - same as teacher but not softmaxed    
      Returns:                                   
          KL divergence loss over the batch                                                         
      """                                           
      assert(teacher_output.shape == student_output.shape)    
                                                     
      student_output_s = F.log_softmax(student_output, dim=2)                      
      teacher_output_s = F.softmax(teacher_output, dim=2)    
                                                                         
      mean_  = 0                                     
      for i in range(teacher_output.shape[0]):                      
          #  looping over samples in a batch                             
          teacher = teacher_output_s[i, :, :]#.unsqueeze(axis=1)    
          student = student_output_s[i, :, :]#.unsqueeze(axis=2)    
          teacher = teacher.unsqueeze(axis=1)                                  
          student = student.unsqueeze(axis=2)                                  
          #ic(teacher.shape, student.shape)                    
          # teacher shape is num x 1 x 80 and student shape is num x 80 x 1    
          cross_entropy_loss = -torch.bmm(teacher, student)                         
          #ic(cross_entropy_loss.shape)    
          #  result is num x 1 x 1                              
                                                 
          #sum_ += sum(cross_entropy_loss)                               
          mean_  += cross_entropy_loss.mean()        
                                                                    
      return mean_/teacher_output.shape[0]        

def old_KL_loss(teacher_output, student_output, id_=1):    
    """Process the KL divergence between the teacher and the student outputs after    
    the sigmoid operation. The logit maps have been normalized for unit norm.    
    
    Arguments:    
        teacher_output (torch.Tensor) - Nx(h*w*num_anchors)x80 for class    
                                      - Nx(h*w*num_anchrs)x4 for reg    
                                      teacher not softmaxed    
        student_output (torch.Tensor) - same as teacher but not softmaxed    
    Returns:    
        KL divergence loss over the batch    
    """    
    assert(teacher_output.shape == student_output.shape)    
    #batch_sum = 0    
    
    student_output_s = F.log_softmax(student_output, dim=2)    
    teacher_output_s = F.softmax(teacher_output, dim=2)    
    
    #for i in range(teacher_output.shape[0]):    
    #    # looping over samples in a batch    
    #    teacher = teacher_output[i, :, :]    
    #    student = student_output[i, :, :]    
    
    ic(teacher_output_s.shape)    
    ic(student_output_s.permute(0,2,1).shape)    
    assert(teacher_output_s.shape == student_output_s.shape)    
    cross_entropy_loss = -torch.bmm(teacher_output_s, student_output_s.permute(0,2,1))    
    cross_entropy_loss = cross_entropy_loss.mean()    
    return cross_entropy_loss 










def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=0, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=1, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="results",
                      type=str)
  parser.add_argument('--exp_name', dest='exp_name',
                      help='experiment folder',
                      type=str)
  parser.add_argument('--caption', dest='caption',
                      help='experiment caption',
                      type=str)
  parser.add_argument('--scheduler', dest='scheduler',
                      help='lr scheduler',
                      type=str)
  parser.add_argument('--server', dest='server',
                      help='experiment server',
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=8, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="adam", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  parser.add_argument('--cdc', dest='cdc',
                      help='classification distillation coeff',
                      default=1, type=float)
  parser.add_argument('--rdc', dest='rdc',
                      help='regression distillation coeff',
                      default=1, type=float)
  args = parser.parse_args()
  return args


def classification_distill_loss(student,  teacher):
    """ student -> input the the classfication head of the RCNN 
        teacher -> input to the classification head of the  RCNN teacher (resnet18)
    """
    ic(student.shape)
    ic(teacher.shape)
    return 1


def regression_distill_loss(reg_student,  reg_teacher):
    """ student -> input the the classfication head of the RCNN 
        teacher -> input to the classification head of the  RCNN teacher (resnet18)
    """
    ic(reg_student.shape)
    ic(reg_teacher.shape)
    return 1


if __name__ == '__main__':

  args = parse_args()
  PARAMS = {'dataset': args.dataset,    
              'exp_name': args.exp_name,    
              'epochs': args.max_epochs,    
              'batch_size': args.batch_size,    
              'lr': args.lr,    
              'caption': args.caption,    
              'server': args.server,
              'scheduler': args.scheduler    
  }    
    
  exp = neptune.create_experiment(name=args.exp_name, params=PARAMS, tags=[args.net,    
                                                                                args.caption,    
                                                                                args.dataset,    
                                                                                args.server])  



  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      #args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      #args.imdbval_name = "coco_2014_minival"
      args.imdb_name = "coco_2017_train"
      args.imdbval_name = "coco_2017_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # Loading the evaluation dataloader 
  #dataloader_eval, imdb_eval = load_evaluation_data()


  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/" + args.exp_name
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  ic(args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)



 

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res18':
    print("loading resnet18")
    fasterRCNN = resnet(imdb.classes, 18, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'birealnet18':
    fasterRCNN = birealnet(imdb.classes, 18, pretrained=False, class_agnostic=args.class_agnostic)
    #fasterRCNN = resnet(imdb.classes, 18, pretrained=False, class_agnostic=args.class_agnostic)

  else:
    print("network is not defined")
    pdb.set_trace()

  distill =  True
  if distill:
      fasterRCNN_teacher = resnet(imdb.classes, 18, pretrained=False, class_agnostic=args.class_agnostic)
        

  fasterRCNN.create_architecture()
  fasterRCNN_teacher.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  if args.optimizer == "adam":
    lr = 0.1 * lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN.cuda()
    fasterRCNN_teacher.cuda()
      
  if args.optimizer == "adam":
    #lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.scheduler=='LambdaLR':
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.max_epochs), last_epoch=-1)

      



  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  # Load student
  student_dir = 'results/birealnet18/coco/adam_Lr0.0001_fasterRCNN_resnet18_baseline_binarynet_LambdaLR_classAgnostic'
  load_name = os.path.join(student_dir, 'faster_rcnn_1_{}_29315.pth'.format(11))
  #student_dir = 'results/birealnet18/coco/adam_Lr0.0001_fasterRCNN_resnet18_baseline_binarynet_RPN_not_binary'
  #teacher_dir = 'results/res18/coco/adam_Lr0.0001_fasterRCNN_resnet18_baseline_full_precision'
  #load_name = os.path.join(teacher_dir, 'faster_rcnn_1_{}_29315.pth'.format(12))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  print('Student Loaded!!')

  # Load teacher
  teacher_dir = 'results/res18/coco/adam_Lr0.0001_fasterRCNN_resnet18_baseline_full_precision'
  load_name = os.path.join(teacher_dir, 'faster_rcnn_1_{}_29315.pth'.format(12))
  checkpoint_teacher = torch.load(load_name)
  fasterRCNN_teacher.load_state_dict(checkpoint_teacher['model'])
  if 'pooling_mode' in checkpoint_teacher.keys():
  # teacher and student should ahve the same pooling. Here, we are setting the student pooling from teacher
  # checkpoint. cfg file already has the POOLING_MODE flag. I think pooling between teacher and student
  # will be consistent regardless of this initialization
    cfg.POOLING_MODE = checkpoint_teacher['pooling_mode']
  fasterRCNN_teacher.eval()
  #fasterRCNN_teacher.train()

  #args.mGPUs=False
  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)
    #fasterRCNN_teacher = nn.DataParallel(fasterRCNN_teacher)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  #set the eval mode for the teacher
  #fasterRCNN_teacher.eval()



  print('teacher loaded')

  #for epoch in range(args.start_epoch, args.max_epochs + 1):
  for epoch in range(args.start_epoch, args.max_epochs):





    exp.log_metric('Current epoch', epoch)
    exp.log_metric('Current lr', float(optimizer.param_groups[0]['lr']))
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()


    '''
    if args.scheduler == 'OldScheduler':
	    if epoch % (args.lr_decay_step + 1) == 0:
		adjust_learning_rate(optimizer, args.lr_decay_gamma)
		lr *= args.lr_decay_gamma
    '''

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])


      with torch.no_grad():
          #fasterRCNN_teacher.eval()
          print('forward on teacher')
          _, cls_prob_teacher, cls_score_teacher, bbox_pred_teacher, \
          rpn_loss_cls_teacher, rpn_loss_box_teacher, \
          RCNN_loss_cls_teacher, RCNN_loss_bbox_teacher, \
          _ = fasterRCNN_teacher(im_data, im_info, gt_boxes, num_boxes)
            
      '''
      loss_teacher = rpn_loss_cls_teacher.mean() + rpn_loss_box_teacher.mean() \
           + RCNN_loss_cls_teacher.mean() + RCNN_loss_bbox_teacher.mean() 


      exp.log_metric('iter_loss_total_teacher', loss_teacher.mean().item())
      exp.log_metric('iter_loss_rpn_cls_teacher', rpn_loss_cls_teacher.mean().item())
      exp.log_metric('iter_loss_rpn_bbox_teacher', rpn_loss_box_teacher.mean().item())
      exp.log_metric('iter_loss_rcnn_cls_teacher', RCNN_loss_cls_teacher.mean().item())
      exp.log_metric('iter_loss_rcnn_bbox_teacher', RCNN_loss_bbox_teacher.mean().item())
      '''

      fasterRCNN.zero_grad()
      print('forward on student')
      rois, cls_prob, cls_score, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      #c_loss_distill = 0    
      reg_loss_distill = 0    
      for i in range(args.batch_size):    
         #class_teacher = cls_score_teacher[i]/ torch.norm(cls_score_teacher[i])    
         reg_teacher = bbox_pred_teacher[i] / torch.norm(bbox_pred_teacher[i])    
         #class_student = cls_score[i] / torch.norm(cls_score[i])    
         reg_student = bbox_pred[i] / torch.norm(bbox_pred[i])    
 
         #c_loss = torch.norm(class_teacher - class_student)    
         r_loss = torch.norm(reg_teacher - reg_student)    

         #c_loss_distill += c_loss    
         reg_loss_distill += r_loss    


      #ic(args.cdc, args.rdc)
      class_distill_loss = args.cdc * KL_loss(cls_score_teacher, cls_score) #(c_loss_distill/args.batch_size)    
      reg_distill_loss = args.rdc * (reg_loss_distill/args.batch_size)    

      #class_distill_loss  = args.cdc * classification_distill_loss(cls_score, cls_score_teacher) 
      #reg_distill_loss = args.rdc  *  regression_distill_loss(bbox_pred, bbox_pred_teacher)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
           + class_distill_loss + reg_distill_loss
      loss_temp += loss.item()


      exp.log_metric('iter_loss_total', loss.mean().item())
      exp.log_metric('iter_loss_rpn_cls', rpn_loss_cls.mean().item())
      exp.log_metric('iter_loss_rpn_bbox', rpn_loss_box.mean().item())
      exp.log_metric('iter_loss_rcnn_cls', RCNN_loss_cls.mean().item())
      exp.log_metric('iter_loss_rcnn_bbox', RCNN_loss_bbox.mean().item())

      exp.log_metric('iter_loss_rcnn_class_distill', class_distill_loss)
      exp.log_metric('iter_loss_rcnn_bbox_distill', reg_distill_loss)



      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f class_distill_loss %.4f reg_distill_loss %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, class_distill_loss, reg_distill_loss))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    if args.scheduler == 'LambdaLR':
        scheduler.step()

    exp.log_metric('epoch_loss_total_teacher', loss_teacher.mean().item())
    exp.log_metric('epoch_loss_rpn_cls_teacher', rpn_loss_cls_teacher.mean().item())
    exp.log_metric('epoch_loss_rpn_bbox_teacher', rpn_loss_box_teacher.mean().item())
    exp.log_metric('epoch_loss_rcnn_cls_teacher', RCNN_loss_cls_teacher.mean().item())
    exp.log_metric('epoch_loss_rcnn_bbox_teacher', RCNN_loss_bbox_teacher.mean().item())

    exp.log_metric('epoch_loss_total', loss.mean().item())
    exp.log_metric('epoch_loss_rpn_cls', rpn_loss_cls.mean().item())
    exp.log_metric('epoch_loss_rpn_bbox', rpn_loss_box.mean().item())
    exp.log_metric('epoch_loss_rcnn_cls', RCNN_loss_cls.mean().item())
    exp.log_metric('epoch_loss_rcnn_bbox', RCNN_loss_bbox.mean().item())
    exp.log_metric('epoch_loss_rcnn_class_distill', class_distill_loss)
    exp.log_metric('epoch_loss_rcnn_bbox_distill', reg_distill_loss)
    
    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))

    # Add evaluation code here
    #evaluate(args, fasterRCNN, dataloader_eval, imdb_eval, output_dir, exp)
    #evaluate(fasterRCNN, dataloader_eval, imdb_eval, output_dir, exp)

  if args.use_tfboard:
    logger.close()







