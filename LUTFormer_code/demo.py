import os
import yaml
import torch
import argparse

from PIL import Image
import torchvision.transforms.functional as TF

from models.networks.get_model import get_model
from utils.viz_utils import Visualizer


def override_config_with_yaml(args, yaml_path):
    if not os.path.exists(yaml_path):
        print(f"[Warning] yaml file {yaml_path} not found. Using default config.")
        return args

    with open(yaml_path, "r") as f:
        override_args = yaml.safe_load(f)

    args_dict = vars(args)
    for k, v in override_args.items():
        args_dict[k] = v

    return argparse.Namespace(**args_dict)


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--input_dir", type=str, default="./demo_img/input", help="directory of input images")
parser.add_argument("--output_dir", type=str, default="./demo_img/result", help="directory of output images")

parser.add_argument("--pretrained_path", type=str, default="./pretrained/Retouching_FiveK_expertC.pth", help="yaml path of pretrained model")
parser.add_argument("--yaml_path", type=str, default="./configs/Retouching_FiveK.yaml", help="yaml path of pretrained model")
parser.add_argument("--task_name", type=str, default="Retouching", help="task name of pretrained model")
parser.add_argument("--dataset_name", type=str, default="FiveK", help="dataset name of pretrained model")
parser.add_argument("--expert", type=str, default="c", help="expert name of pretrained model")

cfg = parser.parse_args()
cfg = override_config_with_yaml(cfg, cfg.yaml_path)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

model = get_model(cfg)
checkpoint = torch.load(cfg.pretrained_path)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
checkpoint = None

model.cuda()
model.eval()

viz_tools = Visualizer(cfg)

with torch.no_grad():
    torch.cuda.empty_cache()

    datalist = os.listdir(cfg.input_dir)
    for img_name in datalist:
        src_name = os.path.join(cfg.input_dir, img_name)
        dst_name = os.path.join(cfg.output_dir, img_name)

        img = Image.open(src_name).convert("RGB")
        img = TF.to_tensor(img)
        img = img[None].cuda()
        outputs = model(img)

        viz_tools.viz_demo(cfg, outputs['enhanced'][0], img_name=img_name)

