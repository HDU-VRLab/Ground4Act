#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import struct
import math
import numpy as np
import cv2
import torch
from scipy import ndimage
from spawn_model import Spawn_object
from collections import OrderedDict
import numpy as np
import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import rospy
import argparse, os,random
import warnings
warnings.filterwarnings("ignore", message="nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.")
warnings.filterwarnings("ignore", message="size_average and reduce args will be deprecated, please use reduction='none' instead.")
warnings.filterwarnings("ignore", message="nn.Upsample is deprecated. Use nn.functional.interpolate instead.")
warnings.filterwarnings("ignore", message="Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0*")

RED = '\033[91m'
PINK = '\033[95m'
BLUE = '\033[94m'
ENDC = '\033[0m'
YELLOW  = '\033[33m'
YELLOW_BOLD   = '\033[33;1m'
GREEN_BOLD = '\033[92;1m'
GREEN = '\033[92m'
def get_outline_from_depth(depth_img):
    mask = np.zeros(depth_img.shape, dtype=np.uint8)   
    mask[depth_img == 255] = 255
    push_binary_segment = mask.copy()
    push_kernel = np.ones((7, 7), np.uint8) 
    push_delete_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    push_mor_segment = cv2.morphologyEx(push_binary_segment, cv2.MORPH_OPEN, push_delete_small)
    push_dilate_kernel = np.ones((15, 15), np.uint8)
    push_dilate_1 = cv2.dilate(push_mor_segment, push_kernel)
    push_dilate_2 = cv2.dilate(push_mor_segment, push_dilate_kernel)
    outline = push_dilate_2 - push_dilate_1
    return mask,outline
def get_model_input(color_heightmap, depth_heightmap, is_volatile=True, specific_rotation=-1):
    color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
    depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
    assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])
    diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)  
    diag_length = np.ceil(diag_length/32)*32 
    padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
    color_heightmap_2x_r = np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
    color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
    color_heightmap_2x_g = np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
    color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
    color_heightmap_2x_b = np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
    color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
    color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
    depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
    image_mean = [0.34751591, 0.29237225, 0.17272882]
    image_std = [0.07591467, 0.08059952, 0.09573705]
    input_color_image = color_heightmap_2x.astype(float)/255
    for c in range(3):
        input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

    image_mean = [0.9791, 0.9791,0.9791]
    image_std = [0.1227, 0.1227, 0.1227]
    cp_depth_heightmap = depth_heightmap_2x.copy()
    cp_depth_heightmap.shape = (cp_depth_heightmap.shape[0], cp_depth_heightmap.shape[1], 1)
    input_depth_image = np.concatenate((cp_depth_heightmap, cp_depth_heightmap, cp_depth_heightmap), axis=2)
    for c in range(3):
        input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

    input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
    input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
    input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
    input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)
    return input_color_data,input_depth_data
    
def policynms_push( push_predictions, num_rotations, valid_depth_heightmap):
    push_action_nms_value = 0
    push_index = 0
    push_final_pixel = None
    for num, push_prediction in enumerate(push_predictions):
        push_mask = np.zeros((224, 224), np.uint8)
        push_pixel = np.unravel_index(np.argmax(push_prediction), push_prediction.shape)
        push_predicted_value = np.max(push_prediction)
        push_roi = [[min(max(push_pixel[1] - 4, 0), 223), min(max(push_pixel[0] + 11, 0), 223)],
                [min(max(push_pixel[1] + 4, 0), 223), min(max(push_pixel[0] + 20, 0), 223)]]
        push_mask = cv2.rectangle(push_mask, (push_roi[0][0], push_roi[0][1]), (push_roi[1][0], push_roi[1][1]), (255, 255, 255), -1)
        push_pixel = tuple(int(x) for x in push_pixel)
        push_rotate = cv2.getRotationMatrix2D((push_pixel[1], push_pixel[0]), num*360/num_rotations, 1)
        push_final = cv2.warpAffine(push_mask, push_rotate, (224, 224))
        push_ROI_multiply = np.zeros(push_final.shape, dtype=np.uint8)
        push_ROI_count = np.zeros(push_final.shape, dtype=np.uint8)
        push_ROI_multiply[push_final >= 100] = 255
        push_ROI_count[push_final >= 100] = 1
        push_total_pixel = np.sum(push_ROI_count)
        push_depth_nms = np.multiply(valid_depth_heightmap, push_ROI_count)
        push_nms_count = np.zeros(push_depth_nms.shape, dtype=np.uint8)
        push_nms_count[push_depth_nms >= 250] = 1
        push_obj_pixel = np.sum(push_nms_count)
        push_prob = float(push_obj_pixel) / (float(push_total_pixel) + 1)
        push_temp = push_prob * push_predicted_value
        if push_temp >= push_action_nms_value:
            push_action_nms_value = push_temp
            push_index = num
            push_final_pixel = push_pixel
    if push_action_nms_value == 0:
        push_locate_part = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        push_predicted_value = np.max(push_predictions)
    else:
        push_locate_part = tuple([push_index, push_final_pixel[0], push_final_pixel[1]])
        push_predicted_value = push_action_nms_value

    return push_locate_part, push_predicted_value

class push_net(nn.Module):
    def __init__(self, use_cuda = True):
        super(push_net, self).__init__()
        self.use_cuda = use_cuda
        self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.push_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.num_rotations = 16
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))
        for m in self.named_modules():
            if 'push-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()
        self.push_interm_feat = []
        self.push_output_prob = []

    def forward(self, input_color_data, input_depth_data, is_volatile=True, specific_rotation=-1):
        if is_volatile:
            with torch.no_grad():
                push_interm_feat = []
                push_output_prob = []
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data).cuda(), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest')
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data), flow_grid_before, mode='nearest')

                    interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                    # print(interm_push_color_feat.shape)  torch.Size([1, 1024, 20, 20])
                    interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                    interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                    push_interm_feat.append(interm_push_feat)
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())
                    
                    push_output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest'))]) 

            return push_output_prob, push_interm_feat

        else:
            self.push_output_prob = []
            self.push_interm_feat = []
            rotate_idx = specific_rotation   
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
            self.push_interm_feat.append(interm_push_feat)
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())
            
            self.push_output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest'))])
            return self.push_output_prob,  self.push_interm_feat