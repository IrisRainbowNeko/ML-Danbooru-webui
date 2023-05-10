import json
import os
import re
from argparse import Namespace
import gradio as gr

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms

from mldanbooru.utils.factory import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def crop_fix(img: Image):
    w, h = img.size
    w = (w//4)*4
    h = (h//4)*4
    return img.crop((0, 0, w, h))

class Infer:
    MODELS = [
        'ml_caformer_m36_fp16_dec-5-97527.ckpt',
        'TResnet-D-FLq_ema_6-30000.ckpt',
    ]
    DEFAULT_MODEL = MODELS[0]
    MODELS_NAME = [
        'caformer_m36',
        'tresnet_d',
    ]
    num_classes = 12547
    RE_SPECIAL = re.compile(r'([\\()])')
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

    def __init__(self):
        self.ca_former_args = Namespace()
        self.ca_former_args.decoder_embedding = 384
        self.ca_former_args.num_layers_decoder = 4
        self.ca_former_args.num_head_decoder = 8
        self.ca_former_args.num_queries = 80
        self.ca_former_args.scale_skip = 1

        self.tresnet_args = Namespace()
        self.tresnet_args.decoder_embedding = 1024
        self.tresnet_args.num_of_groups = 32

        self.args_list = [self.ca_former_args, self.tresnet_args]

        self.last_model_name = None

        self.load_class_map()

    def load_model(self, path=DEFAULT_MODEL):
        ckpt_file = hf_hub_download(repo_id='7eu7d7/ML-Danbooru', filename=path)
        model_idx = self.MODELS.index(path)
        self.model = create_model(self.MODELS_NAME[model_idx], self.num_classes, self.args_list[model_idx]).to(device)
        state = torch.load(ckpt_file, map_location='cpu')
        self.model.load_state_dict(state, strict=True)

    def load_class_map(self):
        classes_file = hf_hub_download(repo_id='7eu7d7/ML-Danbooru', filename='class.json')
        with open(classes_file, 'r') as f:
            self.class_map = json.load(f)

    def build_transform(self, image_size, keep_ratio=False):
        if keep_ratio:
            trans = transforms.Compose([
                transforms.Resize(image_size),
                crop_fix,
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        return trans

    def infer_(self, img: Image.Image, thr: float):
        img = self.trans(img.convert('RGB')).to(device)
        with torch.cuda.amp.autocast():
            img = img.unsqueeze(0)
            output = torch.sigmoid(self.model(img)).cpu().view(-1)
        pred = torch.where(output>thr)[0].numpy()

        cls_list = [(self.class_map[str(i)], output[i]) for i in pred]
        return cls_list

    @torch.no_grad()
    def infer_one(self, img: Image.Image, threshold: float, image_size: int, keep_ratio: bool, model_name: str, space: bool, escape: bool, conf: bool):
        if self.last_model_name != model_name:
            self.load_model(model_name)
            self.last_model_name = model_name

        self.trans = self.build_transform(image_size, keep_ratio)
        cls_list = self.infer_(img, threshold)
        cls_list.sort(reverse=True, key=lambda x:x[1])
        if space:
            cls_list = [(cls.replace('_', ' '), score) for cls, score in cls_list]
        if escape:
            cls_list = [(re.sub(self.RE_SPECIAL, r'\\\1', cls), score) for cls, score in cls_list]

        return ', '.join([f'{cls}:{score:.2f}' if conf else cls for cls, score in cls_list]), {cls:float(score) for cls, score in cls_list}

    @torch.no_grad()
    def infer_folder(self, path: str, threshold: float, image_size: int, keep_ratio: bool, model_name: str, space: bool, escape: bool,
                     out_type: str, prog=gr.Progress()):
        if self.last_model_name != model_name:
            self.load_model(model_name)
            self.last_model_name = model_name

        self.trans = self.build_transform(image_size, keep_ratio)

        tag_dict = {}
        img_list = [os.path.join(path, x) for x in os.listdir(path) if x[x.rfind('.'):].lower() in self.IMAGE_EXTENSIONS]
        for item in prog.tqdm(img_list):
            img = Image.open(item)
            cls_list = self.infer_(img, threshold)
            cls_list.sort(reverse=True, key=lambda x:x[1])
            if space:
                cls_list = [(cls.replace('_', ' '), score) for cls, score in cls_list]
            if escape:
                cls_list = [(re.sub(self.RE_SPECIAL, r'\\\1', cls), score) for cls, score in cls_list]

            if out_type == 'txt':
                with open(item[:item.rfind('.')]+'.txt', 'w', encoding='utf8') as f:
                    f.write(', '.join([name for name, prob in cls_list]))
            elif out_type == 'json':
                tag_dict[os.path.basename(item)] = ', '.join([name for name, prob in cls_list])

        if out_type == 'json':
            with open(os.path.join(path, 'image_captions.json'), 'w', encoding='utf8') as f:
                f.write(json.dumps(tag_dict, indent=2, ensure_ascii=False))

        return 'finish'

    def unload(self):
        if hasattr(self, 'model') and self.model is not None:
            self.last_model_name = None
            del self.model
            return 'model unload'
        return 'no model found'