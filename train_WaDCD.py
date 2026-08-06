import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from collections import OrderedDict
import json
from time import time
from PIL import Image
from copy import deepcopy
from glob import glob
import argparse
import logging
import os
import torch.nn.functional as F
from models import UNET_models
from ldm.modules.diffusionmodules.wavelet_dod_predictor import WaveletDoDPredictor
import random
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from MVTECDataLoader import MVTECDataset
from VISADataLoader import VISADataset
from scipy.ndimage import gaussian_filter
from transformers import get_cosine_schedule_with_warmup


import torch.nn as nn
import math

def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask

def bhpf_highfreq_map(x_img: torch.Tensor, cutoff: int = 40, order: int = 2) -> torch.Tensor:
    """
    Compute Butterworth high-pass filtered grayscale map in [0,1].
    x_img: normalized [-1,1], shape (B,C,H,W)
    return: (B,1,H,W) in [0,1]
    """
    with torch.no_grad():
        B, C, H, W = x_img.shape
        x01 = (x_img * 0.5 + 0.5).clamp(0, 1)
        if C == 3:
            gray = 0.2989 * x01[:, 0] + 0.5870 * x01[:, 1] + 0.1140 * x01[:, 2]
        else:
            gray = x01[:, 0]
        # FFT
        f = torch.fft.fft2(gray)
        fshift = torch.fft.fftshift(f)
        # radial distance grid
        device = x_img.device
        yy = torch.arange(H, device=device).view(H, 1).expand(H, W).float()
        xx = torch.arange(W, device=device).view(1, W).expand(H, W).float()
        crow = (H - 1) / 2.0
        ccol = (W - 1) / 2.0
        dist = torch.sqrt((xx - ccol) ** 2 + (yy - crow) ** 2)
        eps = 1e-6
        mask_hp = 1.0 / (1.0 + (cutoff / (dist + eps)) ** (2 * order))  # high-pass emphasis
        mask_hp = mask_hp.unsqueeze(0)  # (1,H,W)
        fshift_filtered = fshift * mask_hp
        # inverse FFT
        f_ishift = torch.fft.ifftshift(fshift_filtered)
        img_filtered = torch.fft.ifft2(f_ishift).real  # (B,H,W)
        # per-sample min-max normalize to [0,1]
        img_filtered = img_filtered.unsqueeze(1)  # (B,1,H,W)
        b = img_filtered.shape[0]
        v = img_filtered.view(b, -1)
        mn = v.amin(dim=1, keepdim=True).view(b, 1, 1, 1)
        mx = v.amax(dim=1, keepdim=True).view(b, 1, 1, 1)
        out = (img_filtered - mn) / (mx - mn + 1e-8)
        return out

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def shuffle_patches(image, patch_size):
    N, C, H, W = image.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "Image dimensions should be divisible by patch size."

    # Extract patches
    unfolded = F.unfold(image, kernel_size=patch_size, stride=patch_size)  # Shape: (N*C*P*P, num_patches)

    # Reshape unfolded patches to (N, C, P, P, num_patches)
    num_patches = unfolded.shape[-1]
    unfolded = unfolded.view(N, C, P, P, num_patches)

    # Shuffle patches across the batch dimension
    unfolded = unfolded.permute(0, 4, 1, 2, 3)  # Shape: (N, num_patches, C, P, P)
    unfolded = unfolded.reshape(N * num_patches, C, P, P)  # Shape: (N * num_patches, C, P, P)

    # Shuffle patches
    indices = torch.randperm(N * num_patches)
    shuffled_unfolded = unfolded[indices]

    # Reshape back to original format
    shuffled_unfolded = shuffled_unfolded.view(N, num_patches, C, P, P)
    shuffled_unfolded = shuffled_unfolded.permute(0, 2, 3, 4, 1)  # Shape: (N, C, P, P, num_patches)

    # Reconstruct the image
    shuffled_unfolded = shuffled_unfolded.contiguous().view(N * C * P * P, num_patches)
    folded = F.fold(shuffled_unfolded, output_size=(H, W), kernel_size=patch_size, stride=patch_size)

    # Fold operation does not include channels; need to reshape and combine
    folded = folded.view(N, C, H, W)
    
    return folded



def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def random_mask(x : torch.Tensor, mask_ratios, mask_patch_size=1):
    for mask_ratio in mask_ratios:
        assert mask_ratio >=0 and mask_ratio<=1
    n, c, w, h = x.shape
    size = int(np.prod(x.shape[2:]) / (mask_patch_size**2))
    mask = torch.zeros((n,c,size)).to(x.device)
    for b in range(n):
        masked_indexes = np.arange(size)
        np.random.shuffle(masked_indexes)
        masked_indexes = masked_indexes[:int(size * (1 - mask_ratios[b]))]
        mask[b,:, masked_indexes] = 1
    mask = mask.reshape(n, c, int(w/mask_patch_size), int(w/mask_patch_size))
    mask = mask.repeat_interleave(mask_patch_size, dim=2).repeat_interleave(mask_patch_size, dim=3)
    return mask

def set_seed(seed: int, deterministic: bool = True):
    """
    固定 Python、NumPy、PyTorch 随机种子；可选严格确定性。
    """
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 某些算子在严格确定性下可能抛异常，可按需改为 warn_only=True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    set_seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        
        with open(f'{args.results_dir}/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.model_size.replace('/', '-')}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.center_size % 8 == 0, "Image size must be divisible to 8 (for the VAE encoder)."
    latent_size = args.actual_image_size // 8
    in_ch = 5 if getattr(args, 'freq_channel', False) else 4
    base_model = UNET_models[args.model_size](
        latent_size=latent_size,
        ncls=args.num_classes,
        in_channels=in_ch,
    )
    # Wavelet DoD Predictor 外壳：可选 MFFA 和/或 SDEM gate (WGDRB)
    use_mffa = getattr(args, "use_wavelet_mffa", False)
    use_sdem = getattr(args, "use_wavelet_sdem_gate", False)
    if use_mffa or use_sdem:
        model = WaveletDoDPredictor(
            base_model,
            use_mffa=use_mffa,
            use_sdem_gate=use_sdem,
            combine_mode=getattr(args, "wavelet_combine_mode", "sum"),
        )
    else:
        model = base_model
        

    # 预训练加载（鲁棒映射）
    if not args.from_scratch:
        try:
            ckpt = torch.load(args.pretrained, map_location="cpu")
            sd = None
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                    sd = ckpt["state_dict"]
                elif "model" in ckpt and isinstance(ckpt["model"], dict):
                    sd = ckpt["model"]
                else:
                    sd = ckpt
            if not isinstance(sd, dict):
                raise ValueError("checkpoint has no dict-like state_dict/model")
            dictss = {}
            for k, v in sd.items():
                if k.startswith('model.diffusion_model.'):
                    dictss[k.replace('model.diffusion_model.', '')] = v
                elif k.startswith('diffusion_model.'):
                    dictss[k.replace('diffusion_model.', '')] = v
                elif any(k.startswith(p) for p in ('input_blocks', 'middle_block', 'output_blocks', 'out.', 'time_embed')):
                    dictss[k] = v
            # 兼容 ImageNet LDM 的标签嵌入
            if 'model.diffusion_model.label_emb.weight' in sd and 'label_emb.weight' in model.state_dict():
                w = sd['model.diffusion_model.label_emb.weight']
                dictss['label_emb.weight'] = w[:args.num_classes]
            # 兼容 SD 风格 cond_stage embedding（若存在且匹配）
            if 'cond_stage_model.embedding.weight' in sd and 'cond_stage_model.embedding.weight' in model.state_dict():
                w = sd['cond_stage_model.embedding.weight']
                if w.dim() == 2:
                    dictss['cond_stage_model.embedding.weight'] = w[:args.num_classes, :]
            msg = model.load_state_dict(dictss, strict=False)
            logger.info(f"Pretrained loaded: mapped={len(dictss)}, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        except Exception as e:
            logger.info(f"provided pretrained model could not be loaded! Training from scratch ({e})")
                
        

    model = DDP(model.to(device), device_ids=[rank])
    # 根据参数设置扩散步数与respacing（若未显式提供，使用ddim与步数一致）
    respacing = f"ddim{args.diffusion_steps}" if getattr(args, 'timestep_respacing', None) in (None, '') else args.timestep_respacing
    diffusion = create_diffusion(timestep_respacing=respacing, predict_deviation=True, predict_xstart=False, sigma_small=False, diffusion_steps=args.diffusion_steps,
                                 freq_dual_schedule=args.dual_schedule,
                                 noise_schedule_low=args.noise_schedule_low,
                                 noise_schedule_high=args.noise_schedule_high,
                                 lambda_hf=args.lambda_hf,
                                 use_wavelet_loss=args.wavelet_loss,
                                 freq_noise_scale_high=args.freq_noise_scale_high,
                                 wavelet_type=getattr(args, "wavelet_type", "haar"))
    # 离线优先加载 VAE
    project_root = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = []
    if getattr(args, "vae_path", None):
        candidate_dirs.append(args.vae_path)
    candidate_dirs.append(os.path.join(project_root, f"sd-vae-ft-{args.vae_type}"))

    vae = None
    for cand in candidate_dirs:
        if cand and os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
                vae = AutoencoderKL.from_pretrained(cand, local_files_only=True).to(device)
                if rank == 0:
                    logger.info(f"Loaded VAE locally from {cand}")
                break
            except Exception as e:
                if rank == 0:
                    logger.info(f"Failed to load local VAE at {cand}: {e}")
    if vae is None:
        try:
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}").to(device)
            if rank == 0:
                logger.info("Loaded VAE from Hugging Face.")
        except Exception as e:
            raise RuntimeError(f"Cannot load VAE. Provide --vae-path or place sd-vae-ft-{args.vae_type} next to the script. Tried {candidate_dirs}. Error: {e}")
    vae.eval()
    logger.info(f"Number of Parameters: {sum(p.numel() for p in model.parameters()):}")


    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
        
    
    if args.dataset=='mvtec':
        dataset = MVTECDataset('train', object_class=args.object_category, rootdir=args.data_dir, transform=transform, image_size=args.image_size,  center_size=args.center_size, augment=args.augmentation, center_crop=args.center_crop)
    elif args.dataset=='visa':
        dataset = VISADataset('train', object_class=args.object_category, rootdir=args.data_dir, transform=transform, image_size=args.image_size,  center_size=args.center_size, augment=args.augmentation, center_crop=args.center_crop)
    elif args.dataset=='mpdd':
        from MPDDDataLoader import MPDDDataset
        dataset = MPDDDataset('train', object_class=args.object_category, rootdir=args.data_dir, transform=transform, image_size=args.image_size,  center_size=args.center_size, augment=args.augmentation, center_crop=args.center_crop)
       
    batch_size = args.global_batch_size // dist.get_world_size()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    accumulation_steps = args.accumulation_steps

        
    logger.info(f"Dataset contains {len(dataset):,} training images")


    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=args.warmup_epochs,
        num_training_steps=args.epochs*1.5,
    )

    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    running_mse = 0.0
    running_ll = 0.0
    running_hf = 0.0
    running_nan = 0.0
    running_nan_ratio = 0.0
    running_nan_out = 0.0
    running_nan_tgt = 0.0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    
    initial_lambda_hf = args.lambda_hf
    for epoch in range(1, args.epochs+1):
        logger.info(f"Beginning epoch {epoch}...")
        for ii, (x, _, y) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
                if args.freq_channel:
                    # build high-frequency map at image size then downsample to latent size
                    hf = bhpf_highfreq_map(x, cutoff=args.freq_cutoff, order=args.freq_order)
                    # downsample to latent size (H/8,W/8)
                    hf_latent = torch.nn.functional.interpolate(hf, size=(latent.shape[2], latent.shape[3]), mode='area')
                    # map to roughly [-1,1] like latent scaling
                    hf_latent = hf_latent * 2.0 - 1.0
                    x = torch.cat([latent, hf_latent], dim=1)
                else:
                    x = latent
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
 
            if args.actual_image_size == 224:
                mask_patch_size = np.random.choice([1,2,4,8,14], 1, p=[0.3, 0.25, 0.20, 0.15, 0.1]).item()
            elif args.actual_image_size == 256:
                mask_patch_size = np.random.choice([1,2,4,8,16], 1, p=[0.3, 0.25, 0.20, 0.15, 0.1]).item()
            elif args.actual_image_size == 320:
                mask_patch_size = np.random.choice([1,2,4,8,16], 1, p=[0.3, 0.25, 0.20, 0.15, 0.1]).item()
            elif args.actual_image_size == 384:
                mask_patch_size = np.random.choice([1,2,4,8,16,24], 1, p=[0.25, 0.2, 0.20, 0.15, 0.1, 0.1]).item()
            elif args.actual_image_size == 448:
                mask_patch_size = np.random.choice([1,2,4,8,16,28], 1, p=[0.25, 0.2, 0.20, 0.15, 0.1, 0.1]).item()
            elif args.actual_image_size == 512:
                mask_patch_size = np.random.choice([1,2,4,8,16,32], 1, p=[0.25, 0.2, 0.20, 0.15, 0.1, 0.1]).item()   
            if args.mask_random_ratio:
                mask_ratios = np.random.uniform(low=0.0, high=args.mask_ratio, size = x.shape[0])
            else:
                mask_ratio = args.mask_ratio
                mask_ratios = [mask_ratio]*x.shape[0],
                
            mask = random_mask(x, mask_ratios=mask_ratios, mask_patch_size=mask_patch_size)
    
            model_kwargs = {
            'context' : torch.tensor(y).to(device).int().unsqueeze(1),
            'mask': mask
            }
            
            noise_mask = random_mask(x, mask_ratios=np.random.uniform(low=0.0, high=args.patch_shuffle_ratio, size = x.shape[0]), mask_patch_size=mask_patch_size)
            noise = noise_mask * torch.randn_like(x, device=device) + (1-noise_mask) *  shuffle_patches(x, mask_patch_size)
            
            # 高频权重：若启用自动调节，则直接使用当前 args.lambda_hf；否则可选择预热或固定
            if args.wavelet_loss and args.dual_schedule:
                if getattr(args, 'auto_lambda_hf', False):
                    diffusion.lambda_hf = args.lambda_hf
                elif args.hf_warmup_steps > 0 and initial_lambda_hf > 1.0:
                    # 预热：前 hf_warmup_steps 从 1.0 线性升到 initial_lambda_hf
                    warm_prog = min(1.0, train_steps / float(args.hf_warmup_steps))
                    diffusion.lambda_hf = 1.0 + (initial_lambda_hf - 1.0) * warm_prog
                else:
                    # 维持当前（可能经自适应衰减）的值
                    diffusion.lambda_hf = args.lambda_hf
            else:
                diffusion.lambda_hf = args.lambda_hf

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs, noise = noise)
            loss = loss_dict["loss"].mean() / accumulation_steps
            loss.backward()
            
            if (ii + 1) % accumulation_steps == 0:
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad() 
                

            # Log loss values:
            running_loss += loss.item()
            running_mse += loss_dict.get("mse", torch.tensor(0.0, device=device)).mean().item()
            if "loss_ll" in loss_dict:
                running_ll += loss_dict["loss_ll"].mean().item()
            if "loss_hf" in loss_dict:
                running_hf += loss_dict["loss_hf"].mean().item()
            if "nan_detected" in loss_dict:
                running_nan += loss_dict["nan_detected"].float().mean().item()
            if "nan_ratio" in loss_dict:
                running_nan_ratio += loss_dict["nan_ratio"].float().mean().item()
            if "nan_output" in loss_dict:
                running_nan_out += float(loss_dict["nan_output"]) if torch.is_tensor(loss_dict["nan_output"]) else loss_dict["nan_output"]
            if "nan_target" in loss_dict:
                running_nan_tgt += float(loss_dict["nan_target"]) if torch.is_tensor(loss_dict["nan_target"]) else loss_dict["nan_target"]
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_mse = torch.tensor(running_mse / log_steps, device=device)
                avg_ll = torch.tensor(running_ll / max(log_steps,1), device=device)
                avg_hf = torch.tensor(running_hf / max(log_steps,1), device=device)
                avg_nan = torch.tensor(running_nan / max(log_steps,1), device=device)
                avg_nan_ratio = torch.tensor(running_nan_ratio / max(log_steps,1), device=device)
                avg_nan_out = torch.tensor(running_nan_out / max(log_steps,1), device=device)
                avg_nan_tgt = torch.tensor(running_nan_tgt / max(log_steps,1), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_mse, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_ll, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_hf, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_nan, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_nan_ratio, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_nan_out, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_nan_tgt, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_mse = avg_mse.item() / dist.get_world_size()
                avg_ll = avg_ll.item() / dist.get_world_size()
                avg_hf = avg_hf.item() / dist.get_world_size()
                avg_nan = avg_nan.item() / dist.get_world_size()
                avg_nan_ratio = avg_nan_ratio.item() / dist.get_world_size()
                avg_nan_out = avg_nan_out.item() / dist.get_world_size()
                avg_nan_tgt = avg_nan_tgt.item() / dist.get_world_size()
                # 基于 HF/LL 比例的自适应 λ_hf 调节（可选）
                if args.wavelet_loss and args.dual_schedule and getattr(args, 'auto_lambda_hf', False):
                    if avg_ll > 0 and avg_hf > 0:
                        ratio = avg_hf / max(1e-8, avg_ll)
                        # 指数滑动平均，避免抖动
                        if not hasattr(args, "_hf_ratio_ema"):
                            args._hf_ratio_ema = ratio
                        else:
                            args._hf_ratio_ema = args.hf_momentum * args._hf_ratio_ema + (1 - args.hf_momentum) * ratio
                        err = args.hf_target - args._hf_ratio_ema  # 高频占比低时lambda_hf上升
                        # 冷却间隔：在两次自动更新之间至少间隔 auto_cooldown_logs 个日志窗口
                        if not hasattr(args, "_last_auto_update_step"):
                            args._last_auto_update_step = -10**9
                        can_update = (train_steps - args._last_auto_update_step) >= max(1, args.auto_cooldown_logs * args.log_every)
                        # 死区：误差绝对值小于阈值则不更新，避免微小抖动导致持续上升
                        large_err = abs(err) > args.auto_deadband
                        if can_update and large_err:
                            old_lambda = args.lambda_hf
                            # 更新乘子：支持 exp 或 linear 两种模式
                            if args.auto_update_mode == 'exp':
                                mult = math.exp(args.lambda_hf_kp * err)
                            else:
                                # 线性更温和：mult = 1 + kp*err
                                mult = 1.0 + args.lambda_hf_kp * err
                                # 防止非正乘子
                                mult = max(mult, 1e-4)
                            # 乘子夹紧限制，避免单次变化过大
                            mult = float(np.clip(mult, args.auto_mult_min, args.auto_mult_max))
                            args.lambda_hf = float(np.clip(old_lambda * mult, args.lambda_hf_min, args.lambda_hf_max))
                            diffusion.lambda_hf = args.lambda_hf
                            args._last_auto_update_step = train_steps
                            if rank == 0:
                                logger.info(f"[auto] ratio(hf/ll)~{ratio:.3f} (ema {args._hf_ratio_ema:.3f}) target {args.hf_target:.3f} mult {mult:.3f} -> lambda_hf {old_lambda:.3f}->{args.lambda_hf:.3f}")

                # 自适应降低 lambda_hf 防止 NaN（与上面独立；若 NaN 多仍会触发衰减）
                if args.wavelet_loss and args.dual_schedule and args.lambda_hf > 1.0 and avg_nan > 0.3:
                    old_lambda = args.lambda_hf
                    args.lambda_hf = max(1.0, args.lambda_hf * 0.9)
                    # 需要将新值写回 diffusion 实例
                    diffusion.lambda_hf = args.lambda_hf
                    if rank == 0:
                        logger.info(f"High NaN ratio detected ({avg_nan:.2f}); decay lambda_hf: {old_lambda:.3f} -> {args.lambda_hf:.3f}")
                if rank == 0:
                    logger.info(
                        f"(category={args.object_category} step={train_steps:07d}) "
                        f"Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | LL: {avg_ll:.4f} | HF: {avg_hf:.4f} | "
                        f"NaN: {avg_nan:.2f} (ratio {avg_nan_ratio:.2f}, out {avg_nan_out:.2f}, tgt {avg_nan_tgt:.2f}) | "
                        f"lambda_hf: {args.lambda_hf:.3f} | Steps/Sec: {steps_per_sec:.2f}"
                    )
                # Reset monitoring variables:
                running_loss = 0.0
                running_mse = 0.0
                running_ll = 0.0
                running_hf = 0.0
                running_nan = 0.0
                running_nan_ratio = 0.0
                running_nan_out = 0.0
                running_nan_tgt = 0.0
                log_steps = 0
                start_time = time()

        scheduler.step()
        if epoch % args.ckpt_every == 0 and epoch>0:
            if rank == 0: 
                # Save checkpoint:
                checkpoint = {
                    "model": model.module.state_dict(),
                    # "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/epoch-{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
                last_checkpoint_path = f"{checkpoint_dir}/last.pt"
                torch.save(checkpoint, last_checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")

            dist.barrier()
            

    logger.info("Done!")
    cleanup()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['mvtec','visa','mpdd'], default="mvtec")
    parser.add_argument("--data-dir", type=str, default='./mvtec-dataset/')
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_L')
    parser.add_argument("--image-size", type=int, default= 288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=1000)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.7)
    parser.add_argument("--patch-shuffle-ratio", type=float, default=0.3)
    parser.add_argument("--object-category", type=str, default='all')
    parser.add_argument("--mask-random-ratio", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--from-scratch", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--augmentation", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--pretrained", type=str, default='.')
    # 新增：显式指定本地 VAE 目录；若提供将优先离线加载
    parser.add_argument("--vae-path", type=str, default=None, help="本地 VAE 目录（含 config.json 与权重），如 ./sd-vae-ft-ema")
    # 频域通道（L2 in-channels 拼接）
    parser.add_argument("--freq-channel", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=False,
                        help="是否在latent输入拼接一条频域通道（需模型支持5通道输入）")
    parser.add_argument("--freq-cutoff", type=int, default=40, help="BHPF 截止频率")
    parser.add_argument("--freq-order", type=int, default=2, help="BHPF 阶数")
    parser.add_argument("--wavelet-loss", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=False,
                    help="是否启用DWT频率分离MSE")
    parser.add_argument("--lambda-hf", type=float, default=1.0, help="高频loss权重")
    parser.add_argument("--hf-warmup-steps", type=int, default=1000, help="高频权重线性预热步数 (0=禁用)")
        # 自适应 λ_hf 配置
    parser.add_argument("--auto-lambda-hf", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=False,
                        help="是否根据 HF/LL 损失比自动调整 λ_hf（开启后覆盖预热行为）")
    parser.add_argument("--hf-target", type=float, default=1.2, help="期望的 HF/LL 损失比（>1 表示更重高频）")
    parser.add_argument("--lambda-hf-kp", type=float, default=0.001, help="自适应增益（对比误差的比例增益，乘法更新 exp(kp*err)）")
    parser.add_argument("--hf-momentum", type=float, default=0.9, help="HF/LL 比例的 EMA 动量，越大越平滑")
    parser.add_argument("--lambda-hf-min", type=float, default=0.8, help="λ_hf 下限")
    parser.add_argument("--lambda-hf-max", type=float, default=2.0, help="λ_hf 上限")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值 (<=0 不裁剪)")
    parser.add_argument("--dual-schedule", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=False,
                    help="启用低/高频双beta调度")
    parser.add_argument("--noise-schedule-low", type=str, default=None, help="低频beta调度名称，默认与主调度相同")
    parser.add_argument("--noise-schedule-high", type=str, default="squaredcos_cap_v2", help="高频beta调度名称，默认squaredcos_cap_v2")
    parser.add_argument("--freq-noise-scale-high", type=float, default=None, help="不使用双调度时，对高频噪声缩放因子")
    parser.add_argument("--diffusion-steps", type=int, default=20, help="扩散总步数")
    parser.add_argument("--timestep-respacing", type=str, default=None, help="timestep respacing策略，默认使用ddim+步数")
    # Wavelet DoD Predictor 开关
    parser.add_argument("--use-wavelet-mffa", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False,
                        default=False,
                        help="是否在DoD Predictor外壳启用 DWT+MFFA 频域DoD")
    parser.add_argument("--use-wavelet-sdem-gate", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False,
                        default=False,
                        help="是否在 Wavelet DoD Predictor 上叠加 SDEM-style 细节门控")
    parser.add_argument("--wavelet-type", type=str, choices=["haar", "db2"], default="haar",
                        help="小波类型：haar（默认）或 db2（Daubechies-2）")
    parser.add_argument("--wavelet-combine-mode", type=str,
                        choices=["unet-only", "wave-only", "sum"],
                        default="wave-only",
                        help="UNet 与 Wavelet DoD 的组合方式：unet-only / wave-only / sum")
        # 自适应 λ_hf 控制强度：乘子夹紧与冷却间隔
    parser.add_argument("--auto-mult-min", type=float, default=0.98, help="每次自动更新的最小乘子（限制单次下降幅度）")
    parser.add_argument("--auto-mult-max", type=float, default=1.02, help="每次自动更新的最大乘子（限制单次上升幅度）")
    parser.add_argument("--auto-cooldown-logs", type=int, default=1, help="自动更新之间的冷却日志窗口个数（与 log_every 联动）")
    parser.add_argument("--auto-deadband", type=float, default=0.02, help="误差死区阈值；|err|<=deadband 不更新")
    parser.add_argument("--auto-update-mode", type=str, choices=['exp','linear'], default='linear', help="自动更新模式：指数或线性")
    
    args = parser.parse_args()
    if args.dataset == 'mvtec':
        args.num_classes = 15
    elif args.dataset == 'visa':
        args.num_classes = 12
    elif args.dataset == 'mpdd':
        args.num_classes = 6
    args.results_dir = f"./WaDCD_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
    if args.center_crop:
        args.results_dir += "_CenterCrop"
        args.actual_image_size = args.center_size
    else:
        args.actual_image_size = args.image_size
        
    main(args)

