#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
source ~/py3_tf_ws/install/setup.bash --extend
source ~/cv_bridge_ws/install/setup.bash --extend
'''
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from visua_grounding_model import  vl_model
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import sys

from transformers import AutoTokenizer
from transformers import BertTokenizerFast, RobertaTokenizerFast
from RoboRefIt.datasets.refer_segmentation import make_refer_seg_transforms
import rospy
from tools import MyThread,get_camera_points,make_object_info_dict_oneimg,send_pose_to_robot,make_object_info_dict_one_angle,save_image,get_camera_data
from visua_grounding_args import get_args_parser

import torchvision.transforms as transforms
from rl_push_model import push_net,get_model_input,get_outline_from_depth,policynms_push
from spawn_model import Spawn_object
from saveimg import ImageSaver
import random
import math
import warnings
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated since 0.13*")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13*")
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0*")

RED = '\033[91m'
PINK = '\033[95m'
BLUE = '\033[94m'
ENDC = '\033[0m'
YELLOW  = '\033[33m'
YELLOW_BOLD   = '\033[33;1m'
GREEN_BOLD = '\033[92;1m'
GREEN = '\033[92m'

def show_push_pred(rgb_img_show,pix_x, pix_y,angle):
    radius = 5
    color = (0, 0, 255)  
    thickness = 2
    cv2.circle(rgb_img_show, ( pix_x, pix_y ), radius, color, thickness)
    length = 35
    end_x = int(pix_x + length * math.cos(math.radians(-angle)))
    end_y = int(pix_y + length * math.sin(math.radians(-angle)))
    color = (0, 255, 0)  
    thickness = 2
    cv2.arrowedLine(rgb_img_show, ( pix_x, pix_y ), (end_x, end_y), color, thickness)
    cv2.imshow('show_push_pred', rgb_img_show)
    cv2.waitKey(100)

def main(args):
    rospy.init_node('vision_language_push_grasp')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(GREEN+'Loading model...device:{}'.format(device)+ENDC)
    confidence_threshold = 0.15
    # Load the visual grounding net
    vl_net = vl_model(args, device)
    cam_intrinsics = np.array([554.5940455214144, 0.0, 320.5, 0.0, 554.5940455214144, 240.5, 0.0, 0.0, 1.0]).reshape(3,3)
    pred_pix_threshold =  5          
    last_pix_x = 0                  
    last_pix_y = 0                 
    last_angle = 0   
    push_model = push_net(use_cuda = True).to(device)
    push_model_file_path ='src/gjt_ur_moveit_gazebo/env_info/push.pth'
    push_model.load_state_dict(torch.load(push_model_file_path))  
    print(PINK+'Pre-trained push_model loaded from: %s' % (push_model_file_path)+ENDC)
    push_model.eval()
    is_testing = True
    x_min = 0.4-0.1
    x_max = 0.975-0.1
    y_min = -0.2
    y_max = 0.24
    workspace_limits = (x_min-0.1,x_max+0.1,y_min-0.1,y_max+0.1, 0.04, 0.15)
    spawn = Spawn_object(is_testing,workspace_limits,obj_number=6)
    rospy.sleep(2)
    image_saver = ImageSaver()
    image_saver.save_images(color_filename="saved_picture/color{}.png",
                            depth_filename="saved_picture/depth{}.png".format(image_saver.counter))
    print("{} images have been saved".format(image_saver.counter))
    rospy.sleep(1)

    while(True):
        with open('src/picture_counter.txt','r') as f:
            num = int(f.read())
        print(PINK+"Loading the {} image...".format(num+1)+ENDC)
        rgb_path = "saved_picture/color{}.png".format(num)
        depth_path = "saved_picture/depth{}.png".format(num)
        rgb_img = cv2.imread(rgb_path)
        depth = np.array(Image.open(depth_path)).astype(np.float32)
        thread = MyThread(YELLOW+"Please enter the text ('exit' to end the program):"+ENDC)
        thread.start()

        while True:
            cv2.imshow("current image", rgb_img)
            cv2.waitKey(1)
            if not thread.is_alive():
                text = thread.text
                if text == 'exit':
                    print("Exiting,Bye~~")
                    sys.exit()
                print("Get the instruction and start reasoning.....")
                break
        cv2.destroyAllWindows()

        transform_img = make_refer_seg_transforms(args.img_size, args.max_img_size ,test=True, img_type='RGB')
        if rgb_img.shape[-1] > 1:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = np.stack([rgb_img] * 3)
        img = Image.fromarray(rgb_img)
        img, target = transform_img(img, target=None)
        img = img.unsqueeze(0)
        text = text.lower()
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_path)
        tokenized_sentence = tokenizer(
            text,
            padding='max_length',
            max_length=30,
            truncation=True,
            return_tensors='pt',
        )
        word_id = tokenized_sentence['input_ids'][0]
        word_mask = tokenized_sentence['attention_mask'][0]
        word_id = word_id.unsqueeze(0)
        word_mask = word_mask.unsqueeze(0)
        samples = {
            "img": img.to(device).half(),
            "sentence": word_id.to(device),
            "sentence_mask": word_mask.to(device).half(),
            "img_ori": rgb_img
        }
    
        grasp_model_name = []
        grasp_model_z = 0
        Win = False
        inside = True
        vl_net.start_points=[]
        vl_net.end_points=[]
        vl_net.contour_angles=[]
        vl_net.center = []
        bbox, mask,confidence,pred_mask_img,obj_pred_mask = vl_net.forward(samples)
        cv2.imwrite('src/vl_grasp/outputs/save_mask/mask_{}.png'.format(image_saver.counter), obj_pred_mask)
        print(GREEN_BOLD+'confidence:{:.3f},confidence_threshold is {:.3f}'.format(confidence,confidence_threshold)+ENDC)
        # TODO Mask Similarity Comparison（The model always thinks it knows everything）
        if confidence > confidence_threshold :   
            cv2.imshow('show_grasp_pred', pred_mask_img)
            cv2.imwrite('saved_picture/grasp_{}.png'.format(1), pred_mask_img)
            cv2.waitKey(100)
            camera_points=get_camera_points(vl_net.center_x/640*224,vl_net.center_y/480*224,cam_intrinsics)
            object_info_dict = make_object_info_dict_oneimg(camera_points,vl_net.contour_angles)
            send_pose_to_robot(object_info_dict,action_id=1)
            grasp_model_name, grasp_model_z, grasp_model_x, grasp_model_y = spawn.get_grasp_model_name_pos()

        else :
            print(RED+'----------not find ! ready push---------- '+ENDC)
            num_rotations = 16
            color_img, depth_img = get_camera_data()
            color_heightmap = cv2.resize(color_img,(224,224),interpolation = cv2.INTER_LINEAR)
            depth_heightmap = cv2.resize(depth_img,(224,224),interpolation = cv2.INTER_LINEAR)
            mask,outline =  get_outline_from_depth(depth_img)
            outline = cv2.resize(outline,(224,224),interpolation = cv2.INTER_LINEAR)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            push_mor_gradient = cv2.dilate(outline, kernel, iterations=1)
            cv2.imwrite('saved_picture/outline.png',push_mor_gradient)
            cv2.destroyAllWindows()
            input_color_data,input_depth_data = get_model_input(color_heightmap,depth_heightmap)
            print(GREEN_BOLD+'-------push_model.forward-------'+ENDC)
            push_output_prob, _ = push_model.forward(input_color_data, input_depth_data)
            for rotate_idx in range(len(push_output_prob)):
                if rotate_idx == 0:
                    push_predictions = push_output_prob[rotate_idx][0].cpu().data.numpy()[:,0,48:272,48:272]
                else:
                    push_predictions = np.concatenate((push_predictions, push_output_prob[rotate_idx][0].cpu().data.numpy()[:,0,48:272,48:272]), axis=0)
            push_predictions = push_predictions + 0.1
            push_action_mask = np.array(push_mor_gradient.copy()/255).reshape(1,224,224)
            push_predictions = np.multiply(push_predictions, push_action_mask)
            push, push_predicted_value = policynms_push(push_predictions, num_rotations, depth_heightmap)
            '''---------------Compute 3D position of pixel---------[0] - rotate,[1] - pix_y,[2] - pix_x-------------------'''
            best_pix_x = push[2]
            best_pix_y = push[1]
            best_angle = push[0]*(360.0/num_rotations)
            best_rotation_angle = np.deg2rad(best_angle)
            if abs(best_pix_x - last_pix_x) < pred_pix_threshold  and abs(best_pix_y - last_pix_y) < pred_pix_threshold and best_angle == last_angle:
                best_pix_x = best_pix_x + random.randint(-5, 5)
                best_pix_y = best_pix_y + random.randint(-5, 5)
                best_angle = best_angle +  random.choice([-90,-45,45,90])
            last_pix_x = best_pix_x
            last_pix_y = best_pix_y
            last_angle = best_angle
            show_push_pred(color_heightmap,best_pix_x,best_pix_y,best_angle)
            print(YELLOW+'Push pixel position {},{}, Angle {}'.format(best_pix_x,best_pix_y,best_angle)+ENDC)
            angles = []
            angles.append(best_angle)
            camera_points=get_camera_points(best_pix_x,best_pix_y,cam_intrinsics)
            object_info_dict = make_object_info_dict_one_angle(camera_points,angles)
            send_pose_to_robot(object_info_dict,action_id=0)
            
        if grasp_model_z >0.2: 
            print(YELLOW_BOLD+'grasp success!'+ENDC)
            Win = True
            # sys.exit()
            send_pose_to_robot(object_info_dict,action_id=2)
        elif grasp_model_name != [] and grasp_model_z >0.2 : 
            print(RED+'Grasp Error !!'+ENDC)

        inside = spawn.if_obj_in_env()
        print(GREEN_BOLD + ' Objects are in the workspace : {} '.format(inside)+ENDC)
        # new epoch
        if not inside or Win  : 
            spawn.reset_env_from_txt()
            grasp_model_name = []
            grasp_model_z = 0
            rospy.sleep(3)
            image_saver.save_images(color_filename="saved_picture/color{}.png",
                            depth_filename="saved_picture/depth{}.png".format(image_saver.counter))
            print("Saving {} image".format(image_saver.counter))
        else :
            image_saver.save_images(color_filename="saved_picture/color{}.png",
                            depth_filename="saved_picture/depth{}.png".format(image_saver.counter))
            print("Saving {} image".format(image_saver.counter))
        cv2.destroyAllWindows()

if __name__=='__main__':
    parser = argparse.ArgumentParser('Ground4Act', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

