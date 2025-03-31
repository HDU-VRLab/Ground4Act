from .reftr_transformer import build_reftr as build_transformer_based_reftr
from .reftr_segmentation import build_reftr_seg

def build_reftr(args):
    if args.reftr_type.startswith('transformer'):
        if args.masks:
            # use this (transformer_single_phrase,masks=True)
            return build_reftr_seg(args)
        else:
            return build_transformer_based_reftr(args)
    else:
        raise NotImplementedError