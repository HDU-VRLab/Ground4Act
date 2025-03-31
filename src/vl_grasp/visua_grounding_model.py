import os
import sys
sys.path.append('src/vl_grasp/RoboRefIt')
from RoboRefIt.models import build_reftr
from RoboRefIt.util import box_ops

from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2

RED = '\033[91m'
PINK = '\033[95m'
PINK_BOLD = '\033[95;1m'
BLUE = '\033[94m'
ENDC = '\033[0m'
YELLOW  = '\033[33m'
YELLOW_BOLD   = '\033[33;1m'
GREEN_BOLD = '\033[92;1m'
GREEN = '\033[92m'
from tools import get_instance_angle,show_angle,show_img

class vl_model():
    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device
        self.checkpoint = args.checkpoint_vl_path
        self.visualize = True
        self.out_dir = args.output_dir
        self.start_points=[]
        self.end_points=[]
        self.contour_angles=[]
        self.center = []
        self.vl_net,self.criterion, self.postprocessors = self.load_vl_net()

    def load_vl_net(self):
        vl_net, criterion, postprocessors = build_reftr(self.args)
        vl_net.half()
        vl_net.to(self.device)
        checkpoint_vl = torch.load(self.checkpoint)
        vl_net.load_state_dict(checkpoint_vl['model'], strict=False)
        start_epoch = checkpoint_vl['epoch']
        print(PINK+"Pre-trained visual_grounding_model loaded from:  %s (epoch: %d)"%(self.checkpoint, start_epoch)+ENDC)
        vl_net.eval()
        return vl_net, criterion, postprocessors
    
    def process_box(self, bbox, target_size):
        bs, k, _ = bbox.shape
        assert len(bbox) == len(target_size)
        # print("out_bbox.shape:", out_bbox.shape)
        out_bbox = out_bbox[:, 0, :]
        box = box_ops.box_cxcywh_to_xyxy(out_bbox)
        return box
    
    def forward(self, samples):
        if self.visualize:
            output_dir = Path(self.out_dir) / 'vis'
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'mask').mkdir(parents=True, exist_ok=True)
            (output_dir / 'bbox').mkdir(parents=True, exist_ok=True)
            purple = np.array([[[128, 0, 128]]], dtype=np.uint8)
            yellow = np.array([[[255, 255, 0]]], dtype=np.uint8)
                
        img_ori = samples['img_ori']
        outputs = self.vl_net(samples)


        bbox, mask = outputs['pred_boxes'], outputs['pred_masks']
        confidence = outputs['confidence'] 
        bbox = box_ops.box_cxcywh_to_xyxy(bbox[0])

        img_w, img_h = torch.tensor([640]).to(self.device), torch.tensor([480]).to(self.device)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # print(boxes, scale_fct)
        pred_bbox = bbox * scale_fct
        
        if 'segm' in self.postprocessors.keys():
            target_sizes = orig_target_sizes = torch.tensor([[480, 640]]).to(self.device)
            results = [{}]
            results = self.postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = results[0]
            pred_mask_ = res['masks'][0][0].cpu().numpy()
            
            if self.visualize:
                pred_mask = res['masks_origin'][0, 0].cpu().unsqueeze(-1).numpy().astype(np.uint8)
                
                instance_angle ,start_point,end_point = get_instance_angle(pred_mask)
                x = int((start_point[0] + end_point[0])/2)
                y = int((start_point[1] + end_point[1])/2)
                self.center_x = x
                self.center_y = y
                self.contour_angles.append(instance_angle)
                self.start_points.append(start_point)
                self.end_points.append(end_point)

                obj_pred_mask = pred_mask.copy()
                obj_pred_mask = pred_mask*255+ (1-pred_mask)*0
                pred_mask = pred_mask * yellow + (1-pred_mask)*purple
                pred_mask = Image.fromarray(pred_mask)
                pred_mask.save(output_dir / 'mask'/ "0.png")
                pred_box = pred_bbox[0][0].detach().cpu().numpy().tolist()
                
                img_bbox = Image.fromarray(img_ori)
                draw = ImageDraw.Draw(img_bbox)
                draw.rectangle(pred_box, outline='blue', width=5)
                img_bbox.save(output_dir / 'bbox'/ "0.png")

                img_bbox = np.array(img_bbox)
                img_bbox = show_angle(img_bbox,self.start_points,self.end_points,self.contour_angles)
                img_bbox = cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB)
                pred_mask_img = img_bbox.copy()
    
        return pred_box, pred_mask_,confidence,pred_mask_img,obj_pred_mask
        

