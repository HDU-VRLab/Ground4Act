#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('RefTR For Visual Grounding; FGC-GraspNet For Grasp Pose Detction', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["img_backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_mask_branch_names', default=['bbox_attention', 'mask_head'], type=str, nargs='+')
    parser.add_argument('--lr_mask_branch_proj', default=1., type=float)
    parser.add_argument('--lr_bert_names', default=["lang_backbone"], type=str, nargs='+')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--warm_up_epoch', default=2, type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--lr_schedule', default='StepLR', type=str)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--ckpt_cycle', default=20, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--no_decoder', default=False, action='store_true')
    parser.add_argument('--reftr_type', default='transformer_single_phrase', type=str,
                        help="using bert based reftr vs transformer based reftr")
    parser.add_argument('--pretrain_on_coco', default=False, action='store_true')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help="Path to the pretrained model. If set, DETR weight will be used to initilize the network.")
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--ablation', type=str, default='none', help="Ablation")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--masks', default=True,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--freeze_reftr', action='store_true',
                        help="Train unfreeze reftr for segmentation if the flag is provided")

    parser.add_argument('--bert_model', default="bert-base-uncased", type=str,
                        help="bert model name for transformer based reftr")
    parser.add_argument('--img_bert_config', default="./configs/VinVL_VQA_base", type=str,
                        help="For bert based reftr: Path to default image bert ")
    parser.add_argument('--use_encoder_pooler', default=False, action='store_true',
                        help="For bert based reftr: Whether to enable encoder pooler ")
    parser.add_argument('--freeze_bert', action='store_true',
                        help="Whether to freeze language bert")
    parser.add_argument('--max_lang_seq', default=128, type=int,
                        help="Controls maxium number of embeddings in VLTransformer")
    parser.add_argument('--num_queries_per_phrase', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--aux_loss', action='store_true',
                        help="Enable auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_softmax_ce', action='store_true',
                        help="Whether to use cross entropy loss over all queries")
    parser.add_argument('--bbox_loss_topk', default=1, type=int,
                        help="set > 1 to enbale softmargin loss and topk picking in vg loss ")

    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=1, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--dataset', default='roborefit')
    parser.add_argument('--data_root', default='data/final_dataset')
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--test_split', default='testA', type=str)
    parser.add_argument('--img_size', default=640, type=int)
    parser.add_argument('--img_type', default='RGB')
    parser.add_argument('--max_img_size', default=640, type=int)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/mscoco', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
 
    parser.add_argument('--image_path', default='example/image')
    parser.add_argument('--depth_path', default='example/depth')
    parser.add_argument('--text_path', default='example/text')
    parser.add_argument('--checkpoint_vl_path', default='src/vl_grasp/logs/visua_grounding.pth')
    parser.add_argument('--bert_model_path',default='src/vl_grasp/logs/bert-base-uncased')
    parser.add_argument('--output_dir', default='src/vl_grasp/outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--resume_model_only', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--run_epoch', default=500, type=int, metavar='N',
                        help='epochs for current run')                
    parser.add_argument('--eval', default=False)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    
    return parser