import torch
from skimage.transform import resize
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import UNET_models
from ldm.modules.diffusionmodules.wavelet_dod_predictor import WaveletDoDPredictor
import argparse
import numpy as np
torch.set_grad_enabled(False)
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
from glob import glob
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from MVTECDataLoader import MVTECDataset
from VISADataLoader import VISADataset
from scipy.ndimage import gaussian_filter
import torch

from anomalib import metrics
from sklearn.metrics import average_precision_score
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import auc

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": [np.mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc




def calculate_metrics(ground_truth, prediction):
    flat_gt = ground_truth.flatten()
    flat_pred = prediction.flatten()
    

    auprc = metrics.AUPR()
    auprc_score = auprc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    # aupro_score = 0
    aupro = metrics.AUPRO(fpr_limit=0.3)
    aupro_score = compute_pro(ground_truth, prediction)
    
    auroc = metrics.AUROC()
    auroc_score = auroc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    f1max = metrics.F1Max()
    f1_max_score = f1max(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))
    
    ap = average_precision_score(ground_truth.flatten(), prediction.flatten())
    
    gt_list_sp = []
    pr_list_sp = []
    for idx in range(len(ground_truth)):
        gt_list_sp.append(np.max(ground_truth[idx]))
        sp_score = np.max(prediction[idx])
        pr_list_sp.append(sp_score)

    gt_list_sp = np.array(gt_list_sp).astype(np.int32)
    pr_list_sp = np.array(pr_list_sp)

    apsp = average_precision_score(gt_list_sp, pr_list_sp)
    aurocsp = auroc(torch.from_numpy(pr_list_sp), torch.from_numpy(gt_list_sp))
    f1sp = f1max(torch.from_numpy(pr_list_sp), torch.from_numpy(gt_list_sp))
    
    return auroc_score.numpy(), aupro_score ,f1_max_score.numpy(), ap, aurocsp.numpy(), apsp, f1sp.numpy()


def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask

def bhpf_highfreq_map(x_img: torch.Tensor, cutoff: int = 40, order: int = 2) -> torch.Tensor:
    """
    Butterworth 高通灰度图，输出 [0,1]，形状 (B,1,H,W)。x_img 输入为 [-1,1] 归一化。
    """
    with torch.no_grad():
        B, C, H, W = x_img.shape
        x01 = (x_img * 0.5 + 0.5).clamp(0, 1)
        if C == 3:
            gray = 0.2989 * x01[:, 0] + 0.5870 * x01[:, 1] + 0.1140 * x01[:, 2]
        else:
            gray = x01[:, 0]
        f = torch.fft.fft2(gray)
        fshift = torch.fft.fftshift(f)
        device = x_img.device
        yy = torch.arange(H, device=device).view(H, 1).expand(H, W).float()
        xx = torch.arange(W, device=device).view(1, W).expand(H, W).float()
        crow = (H - 1) / 2.0
        ccol = (W - 1) / 2.0
        dist = torch.sqrt((xx - ccol) ** 2 + (yy - crow) ** 2)
        eps = 1e-6
        mask_hp = 1.0 / (1.0 + (cutoff / (dist + eps)) ** (2 * order))
        mask_hp = mask_hp.unsqueeze(0)
        fshift_filtered = fshift * mask_hp
        f_ishift = torch.fft.ifftshift(fshift_filtered)
        img_filtered = torch.fft.ifft2(f_ishift).real
        img_filtered = img_filtered.unsqueeze(1)
        b = img_filtered.shape[0]
        v = img_filtered.view(b, -1)
        mn = v.amin(dim=1, keepdim=True).view(b, 1, 1, 1)
        mx = v.amax(dim=1, keepdim=True).view(b, 1, 1, 1)
        out = (img_filtered - mn) / (mx - mn + 1e-8)
        return out


    

def calculate_anomaly_maps(x0_s, encoded_s,  image_samples_s, latent_samples_s, center_size=256):
    pred_geometric = []
    pred_aritmetic = []
    image_differences = []
    latent_differences = []
    input_images = []
    output_images = []
    for x, encoded,  image_samples, latent_samples in zip(x0_s, encoded_s,  image_samples_s, latent_samples_s):
            
        input_image = ((np.clip(x[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
        output_image = ((np.clip(image_samples[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
        input_images.append(input_image)
        output_images.append(output_image)

        image_difference = (((((torch.abs(image_samples-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
        image_difference = (np.clip(image_difference, 0.0, 0.4) ) * 2.5
        image_difference = smooth_mask(image_difference, sigma=3)
        image_differences.append(image_difference)
        
        # 对齐通道数：当启用频域通道时 latent_samples 可能为 5c，而 encoded 为 4c，这里仅比较前 4 个 VAE 通道
        if latent_samples.shape[1] != encoded.shape[1]:
            latent_samples_cmp = latent_samples[:, :encoded.shape[1], ...]
        else:
            latent_samples_cmp = latent_samples
        latent_difference = (((((torch.abs(latent_samples_cmp - encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
        latent_difference = (np.clip(latent_difference, 0.0 , 0.2)) * 5
        latent_difference = smooth_mask(latent_difference, sigma=1)
        latent_difference = resize(latent_difference, (center_size, center_size))
        latent_differences.append(latent_difference)
        
        final_anomaly = image_difference * latent_difference
        final_anomaly = np.sqrt(final_anomaly)
        final_anomaly2 = 1/2*image_difference + 1/2*latent_difference
        pred_geometric.append(final_anomaly)
        pred_aritmetic.append(final_anomaly2)
            
    pred_geometric = np.stack(pred_geometric, axis=0)
    pred_aritmetic = np.stack(pred_aritmetic, axis=0)
    latent_differences = np.stack(latent_differences, axis=0)
    image_differences = np.stack(image_differences, axis=0)

    return {'anomaly_geometric':pred_geometric, 'anomaly_aritmetic':pred_aritmetic, 'latent_discrepancy':latent_differences, 'image_discrepancy':image_differences}



def evaluate_anomaly_maps(anomaly_maps, segmentation):
    results = {}
    for key in anomaly_maps.keys():
        auroc_score, aupro_score ,f1_max_score, ap, aurocsp, apsp, f1sp = calculate_metrics(segmentation, anomaly_maps[key])
        vals = {
            "P-AUROC": float(auroc_score),
            "P-AUPRO": float(aupro_score),
            "P-f1max": float(f1_max_score),
            "P-AP": float(ap),
            "I-AUROC": float(aurocsp),
            "I-AP": float(apsp),
            "I-f1max": float(f1sp),
        }
        results[key] = vals
        print('{}: auroc:{:.4f}, aupro:{:.4f}, f1_max:{:.4f}, ap:{:.4f}, aurocsp:{:.4f}, apsp:{:.4f}, f1sp:{:.4f}'.format(
            key, vals["P-AUROC"], vals["P-AUPRO"], vals["P-f1max"], vals["P-AP"], vals["I-AUROC"], vals["I-AP"], vals["I-f1max"]
        ))
    return results


def evaluation(args):
    # 离线优先加载 VAE；若本地不存在再回退到 Hugging Face
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
                print(f"Loaded VAE locally from {cand}")
                break
            except Exception as e:
                print(f"Failed to load local VAE at {cand}: {e}")
    if vae is None:
        vae_model = f"stabilityai/sd-vae-ft-{args.vae_type}"  # will try online if offline not available
        vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    vae.eval()

    try:
        if args.model_path != '':
            ckpt = args.model_path
        else:
            path = f"./WaDCD_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
            try:
                ckpt = sorted(glob(f'{path}/last.pt'))[-1]
            except:
                ckpt = sorted(glob(f'{path}/*/last.pt'))[-1]
    except:
        raise Exception("Please provide the trained model's path using --model_path")
    

    latent_size = int(args.center_size) // 8
    in_ch = 5 if getattr(args, 'freq_channel', False) else 4
    base_model = UNET_models[args.model_size](
        latent_size=latent_size,
        ncls=args.num_classes,
        in_channels=in_ch,
    )
    if getattr(args, "use_wavelet_mffa", False):
        model = WaveletDoDPredictor(
            base_model,
            use_mffa=True,
            use_sdem_gate=getattr(args, "use_wavelet_sdem_gate", False),
            combine_mode=getattr(args, "wavelet_combine_mode", "wave-only"),
        )
    else:
        model = base_model

    # : 
    #  wavelet  MFFA/SDEM  state_dict 
    # 1)  ckpt 
    ckpt_state = torch.load(ckpt)["model"]
    model_state = model.state_dict()

    # 2) :  key  shape 
    filtered_state = {}
    skipped_keys = []
    for k, v in ckpt_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            skipped_keys.append(k)

    # 3)  state_dict 
    model_state.update(filtered_state)
    msg = model.load_state_dict(model_state, strict=False)
    print("[WaDCD Eval] load_state_dict -> missing:", len(msg.missing_keys), "unexpected:", len(msg.unexpected_keys))
    if len(skipped_keys) > 0:
        print("[WaDCD Eval] skipped keys from checkpoint (shape mismatch / not in model):", len(skipped_keys))
        # mffa.*  key 
        sample_skipped = [k for k in skipped_keys if k.startswith("mffa.")][:8]
        if sample_skipped:
            print("  e.g. ", sample_skipped)

    model.eval()  # important!
    model.to(device)
    print('model loaded')


    print('=='*30)
    print('Starting Evaluation...')
    print('=='*30)

    for category in args.categories:


        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
            
        # Create diffusion object: 与训练保持一致的步数与调度
        respacing = args.timestep_respacing if getattr(args, 'timestep_respacing', None) else f'ddim{args.reverse_steps}'
        diffusion = create_diffusion(
            timestep_respacing=respacing,
            predict_deviation=True,
            sigma_small=False,
            predict_xstart=False,
            diffusion_steps=args.diffusion_steps,
            freq_dual_schedule=args.dual_schedule,
            noise_schedule_low=args.noise_schedule_low,
            noise_schedule_high=args.noise_schedule_high,
            dual_schedule_sampling=args.dual_schedule_sampling,
        )
            

        encoded_s = []
        image_samples_s = []
        latent_samples_s = []
        x0_s = []
        segmentation_s = []
        
        if args.dataset=='mvtec':
            test_dataset = MVTECDataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop)
        else:
            test_dataset = VISADataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)
        
        for ii, (x, seg, object_cls) in enumerate(test_loader):
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = x.to(device)
                encoded = vae.encode(x).latent_dist.mean.mul_(0.18215)
                if getattr(args, 'freq_channel', False):
                    hf = bhpf_highfreq_map(x, cutoff=args.freq_cutoff, order=args.freq_order)
                    hf_latent = F.interpolate(hf, size=(encoded.shape[2], encoded.shape[3]), mode='area')
                    hf_latent = hf_latent * 2.0 - 1.0
                    encoded_aug = torch.cat([encoded, hf_latent], dim=1)
                else:
                    encoded_aug = encoded
                model_kwargs = {
                'context':object_cls.to(device).unsqueeze(1),
                'mask': None
                }
                latent_samples = diffusion.ddim_deviation_sample_loop(
                    model, encoded_aug.shape, noise = encoded_aug, clip_denoised=False, 
                    start_t = args.reverse_steps,
                    model_kwargs=model_kwargs, progress=False, device=device,
                    eta = 0
                )

                # 仅解码前4通道
                image_samples = vae.decode(latent_samples[:, :4] / 0.18215).sample 
                x0 = vae.decode(encoded[:, :4] / 0.18215).sample 

            segmentation_s += [_seg.squeeze() for _seg in seg]
            encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
            image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_samples]
            x0_s += [_x0.unsqueeze(0) for _x0 in x0]

        print(category)        
        anomaly_maps = calculate_anomaly_maps(x0_s, encoded_s,  image_samples_s, latent_samples_s, center_size=args.center_size)
        metrics_dict = evaluate_anomaly_maps(anomaly_maps, np.stack(segmentation_s, axis=0))
        # 记录本类用于汇总的指标（从指定 report_key 选择）；默认 anomaly_geometric
        if not hasattr(args, 'dataset_summary'):
            args.dataset_summary = []
        report_key = getattr(args, 'report_key', 'anomaly_geometric')
        args.dataset_summary.append(metrics_dict[report_key])
        print('=='*30)  

    # 类别循环结束后，做数据集级别汇总（宏平均 across categories），并可保存CSV
    if getattr(args, 'dataset_summary', None):
        mean = lambda k: float(np.mean([m[k] for m in args.dataset_summary]))
        I_AUROC = mean("I-AUROC"); I_AP = mean("I-AP"); I_F1 = mean("I-f1max")
        P_AUROC = mean("P-AUROC"); P_AP = mean("P-AP"); P_F1 = mean("P-f1max"); P_AUPRO = mean("P-AUPRO")
        print("Dataset summary (macro over {} categories) using {}".format(len(args.dataset_summary), report_key))
        print("I-AUROC:{:.1f}  I-AP:{:.1f}  I-f1max:{:.1f}  P-AUROC:{:.1f}  P-AP:{:.1f}  P-f1max:{:.1f}  P-AUPRO:{:.1f}".format(
            I_AUROC*100, I_AP*100, I_F1*100, P_AUROC*100, P_AP*100, P_F1*100, P_AUPRO*100
        ))
        # 保存 CSV
        save_path = args.save_summary if getattr(args, 'save_summary', '') else os.path.join(args.results_dir, f"summary_{report_key}.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import csv
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value(%)"]) 
            writer.writerow(["I-AUROC", f"{I_AUROC*100:.2f}"])
            writer.writerow(["I-AP", f"{I_AP*100:.2f}"])
            writer.writerow(["I-f1max", f"{I_F1*100:.2f}"])
            writer.writerow(["P-AUROC", f"{P_AUROC*100:.2f}"])
            writer.writerow(["P-AP", f"{P_AP*100:.2f}"])
            writer.writerow(["P-f1max", f"{P_F1*100:.2f}"])
            writer.writerow(["P-AUPRO", f"{P_AUPRO*100:.2f}"])
        print(f"Saved summary to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['mvtec','visa'], default="mvtec")
    parser.add_argument("--data-dir", type=str, default='./mvtec-dataset/')
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_L')
    parser.add_argument("--image-size", type=int, default= 288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--object-category", type=str, default='all')
    parser.add_argument("--model-path", type=str, default='.')
    parser.add_argument("--anomaly-class", type=str, default='all')
    parser.add_argument("--reverse-steps", type=int, default=20)
    # 扩散/调度相关（与训练保持一致用法）
    parser.add_argument("--diffusion-steps", type=int, default=20, help="扩散总步数，应 >= reverse-steps")
    parser.add_argument("--timestep-respacing", type=str, default=None, help="自定义respacing，如 ddim20；缺省按 reverse-steps 生成")
    parser.add_argument("--dual-schedule", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=False, help="启用低/高频双beta调度")
    parser.add_argument("--noise-schedule-low", type=str, default=None, help="低频beta调度名称，默认与主调度相同")
    parser.add_argument("--noise-schedule-high", type=str, default="squaredcos_cap_v2", help="高频beta调度名称，默认squaredcos_cap_v2")
    parser.add_argument("--dual-schedule-sampling", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=False, help="推理阶段是否启用双频回溯")
    # 频域通道（与训练保持一致）
    parser.add_argument("--freq-channel", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=False)
    parser.add_argument("--freq-cutoff", type=int, default=40)
    parser.add_argument("--freq-order", type=int, default=2)
    # 新增：选择用于数据集汇总的异常图键
    parser.add_argument("--report-key", type=str, choices=['anomaly_geometric','anomaly_aritmetic','latent_discrepancy','image_discrepancy'], default='anomaly_geometric')
    # 新增：宏平均结果保存路径（CSV）。留空则保存到 results_dir/summary_{report_key}.csv
    parser.add_argument("--save-summary", type=str, default="")
    # 新增：显式指定本地 VAE 目录（包含 config.json 与权重文件），优先离线加载
    parser.add_argument("--vae-path", type=str, default=None)
    # Wavelet DoD Predictor 开关：与训练端一致
    parser.add_argument("--use-wavelet-mffa", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False,
                        default=False,
                        help="是否在DoD Predictor外壳启用 DWT+MFFA 频域DoD")
    parser.add_argument("--use-wavelet-sdem-gate", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False,
                        default=False,
                        help="是否在 Wavelet DoD Predictor 上叠加 SDEM-style 细节门控")
    parser.add_argument("--wavelet-combine-mode", type=str,
                        choices=["unet-only", "wave-only", "sum"],
                        default="wave-only",
                        help="UNet 与 Wavelet DoD 的组合方式：unet-only / wave-only / sum")
    
    args = parser.parse_args()
    if args.dataset == 'mvtec':
        args.num_classes = 15
    elif args.dataset == 'visa':
        args.num_classes = 12
    args.results_dir = f"./WaDCD_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
    if args.center_crop:
        args.results_dir += "_CenterCrop"
        args.actual_image_size = args.center_size
    else:
        args.actual_image_size = args.image_size

    if args.object_category=='all' and args.dataset=='mvtec':
        args.categories=[
            "bottle",
            "cable",
            "capsule",
            "hazelnut",
            "metal_nut",
            "pill",
            "screw",
            "toothbrush",
            "transistor",
            "zipper",
            "carpet",
            "grid",
            "leather",
            "tile",
            "wood",
            ]
    elif args.object_category=='all' and args.dataset=='visa':
        args.categories=[
            "candle",
            "cashew",
            "fryum",
            "macaroni2",
            "pcb2",
            "pcb4",
            "capsules",
            "chewinggum",
            "macaroni1",
            "pcb1",
            "pcb3",
            "pipe_fryum"
            ]
    else:
        args.categories = [args.object_category]
        
    evaluation(args)
