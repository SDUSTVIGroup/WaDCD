import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.transform import resize
from PIL import Image

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import UNET_models


def _load_vae(device, vae_type="ema", vae_path=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_dirs = []
    if vae_path:
        candidate_dirs.append(vae_path)
    candidate_dirs.append(os.path.join(project_root, f"sd-vae-ft-{vae_type}"))
    vae = None
    for cand in candidate_dirs:
        if cand and os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
                vae = AutoencoderKL.from_pretrained(cand, local_files_only=True).to(device)
                print(f"Loaded VAE locally from {cand}")
                break
            except Exception as e:
                print(f"Failed to load local VAE at {cand}: {e}")
    if vae is None:
        from diffusers.models import AutoencoderKL as _Auto
        vae = _Auto.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
    vae.eval()
    return vae


def _build_model(ckpt_path, model_size, latent_size, device):
    model = UNET_models[model_size](latent_size=latent_size, ncls=15)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def _calc_maps(x0_s, encoded_s,  image_samples_s, latent_samples_s, center_size=256, mode='anomaly_geometric'):
    pred_geometric = []
    pred_aritmetic = []
    for x, encoded,  image_samples, latent_samples in zip(x0_s, encoded_s,  image_samples_s, latent_samples_s):
        image_difference = (((((torch.abs(image_samples-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
        image_difference = (np.clip(image_difference, 0.0, 0.4) ) * 2.5
        latent_difference = (((((torch.abs(latent_samples-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
        latent_difference = (np.clip(latent_difference, 0.0 , 0.2)) * 5
        latent_difference = resize(latent_difference, (center_size, center_size))
        final_anomaly = image_difference * latent_difference
        final_anomaly = np.sqrt(final_anomaly)
        final_anomaly2 = 0.5*image_difference + 0.5*latent_difference
        pred_geometric.append(final_anomaly)
        pred_aritmetic.append(final_anomaly2)
    pred_geometric = np.stack(pred_geometric, axis=0)
    pred_aritmetic = np.stack(pred_aritmetic, axis=0)
    return pred_geometric if mode=='anomaly_geometric' else pred_aritmetic


def export_mvtec(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # transform 与评估一致
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # 读取 split CSV
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'splits', 'mvtec-split.csv')
    df = pd.read_csv(csv_path)
    if args.category != 'all':
        df = df.query('split=="test" and object==@args.category')
    else:
        df = df.query('split=="test"')
    if len(df)==0:
        raise RuntimeError('No test samples found in mvtec-split.csv')

    # build model/diffusion/vae
    vae = _load_vae(device, vae_type=args.vae_type, vae_path=args.vae_path)
    latent_size = args.center_size // 8
    model = _build_model(args.model_path, args.model_size, latent_size, device)
    diffusion = create_diffusion(f'ddim{args.reverse_steps}', predict_deviation=True, sigma_small=False, predict_xstart=False, diffusion_steps=10)

    # 按 object 分组，分小批次处理
    objects = sorted(df['object'].unique()) if args.category=='all' else [args.category]
    for obj in objects:
        df_obj = df[df['object']==obj]
        # 文件路径列表
        img_relpaths = df_obj['image'].tolist()
        # 按批读取图像
        batch = []
        paths = []
        for rel in img_relpaths:
            abspath = os.path.join(args.data_dir, rel)
            img = Image.open(abspath).convert('RGB')
            # 可选中心裁剪到 center_size
            if args.center_crop:
                # 假设输入是方形 resize 到 image_size 再 center crop
                img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
                dx = (args.image_size - args.center_size) // 2
                box = (dx, dx, dx+args.center_size, dx+args.center_size)
                img = img.crop(box)
            else:
                img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
            x = transform(np.array(img)/255.0)
            batch.append(x)
            paths.append(rel)
            if len(batch) == args.batch_size:
                _process_and_save(batch, paths, vae, model, diffusion, device, args)
                batch, paths = [], []
        if batch:
            _process_and_save(batch, paths, vae, model, diffusion, device, args)


def _process_and_save(batch, paths, vae, model, diffusion, device, args):
    x = torch.stack(batch, dim=0).to(device)
    with torch.no_grad():
        encoded = vae.encode(x).latent_dist.mean.mul_(0.18215)
        model_kwargs = { 'context': torch.zeros(len(batch),1, device=device, dtype=torch.long), 'mask': None }
        latent_samples = diffusion.ddim_deviation_sample_loop(
            model, encoded.shape, noise=encoded, clip_denoised=False,
            start_t=args.reverse_steps, model_kwargs=model_kwargs,
            progress=False, device=device, eta=0
        )
        image_samples = vae.decode(latent_samples / 0.18215).sample

    # 组装为单张并计算 map
    x0_s = [x[i].unsqueeze(0) for i in range(x.shape[0])]
    enc_s = [encoded[i].unsqueeze(0) for i in range(encoded.shape[0])]
    img_s = [image_samples[i].unsqueeze(0) for i in range(image_samples.shape[0])]
    lat_s = [latent_samples[i].unsqueeze(0) for i in range(latent_samples.shape[0])]

    maps = _calc_maps(x0_s, enc_s, img_s, lat_s, center_size=args.center_size, mode=args.map_key)
    # 归一化到 0-255 并保存到 obj/anomaly/filename.png
    for i, rel in enumerate(paths):
        # rel: e.g., bottle/test/broken_large/000.png
        parts = rel.split('/')
        if len(parts) < 4:
            # 回退：保存到 flat 结构
            out_path = os.path.join(args.output_dir, os.path.basename(rel))
        else:
            obj = parts[0]; split = parts[1]; anomaly = parts[2]; name = parts[3]
            out_dir = os.path.join(args.output_dir, obj, anomaly)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, name)
        amap = maps[i]
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        Image.fromarray((amap*255).astype(np.uint8)).save(out_path)


def main():
    parser = argparse.ArgumentParser("Export WaDCD anomaly maps for AnomalyNCD")
    parser.add_argument('--dataset', type=str, choices=['mvtec'], default='mvtec')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--category', type=str, default='all')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--model-size', type=str, choices=['UNet_XS','UNet_S','UNet_M','UNet_L','UNet_XL'], default='UNet_L')
    parser.add_argument('--image-size', type=int, default=288)
    parser.add_argument('--center-size', type=int, default=256)
    parser.add_argument('--center-crop', type=lambda v: True if str(v).lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument('--reverse-steps', type=int, default=5)
    parser.add_argument('--vae-type', type=str, choices=['ema','mse'], default='ema')
    parser.add_argument('--vae-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--map-key', type=str, choices=['anomaly_geometric','anomaly_aritmetic'], default='anomaly_geometric')
    parser.add_argument('--output-dir', type=str, required=True, help='e.g., data/mvtec_musc_anomaly_map')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset == 'mvtec':
        export_mvtec(args)
    else:
        raise NotImplementedError('Only mvtec is supported in this exporter v1')


if __name__ == '__main__':
    main()
