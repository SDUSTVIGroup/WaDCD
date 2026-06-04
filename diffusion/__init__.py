# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="squaredcos_cap_v2",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    predict_deviation=True,
    learn_sigma=False,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    # 新增：频域双调度/损失
    freq_dual_schedule: bool = False,
    noise_schedule_low: str | None = None,
    noise_schedule_high: str | None = None,
    lambda_hf: float = 1.0,
    use_wavelet_loss: bool = False,
    freq_noise_scale_high: float | None = None,
    # 新增：推理阶段是否按双调度进行频域反向
    dual_schedule_sampling: bool = False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    betas_low = None
    betas_high = None
    if freq_dual_schedule:
        # 若未指定，则默认沿用主schedule
        low_name = noise_schedule_low or noise_schedule
        high_name = noise_schedule_high or noise_schedule
        betas_low = gd.get_named_beta_schedule(low_name, diffusion_steps)
        betas_high = gd.get_named_beta_schedule(high_name, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    if predict_xstart:
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_deviation:
        model_mean_type = gd.ModelMeanType.DEVIATION
    else:
        model_mean_type = gd.ModelMeanType.EPSILON
        
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
          model_mean_type
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        # 传递频域附加配置（SpacedDiffusion会下传到GaussianDiffusion）
        betas_low=betas_low,
        betas_high=betas_high,
        lambda_hf=lambda_hf,
        use_wavelet_loss=use_wavelet_loss,
        freq_noise_scale_high=freq_noise_scale_high,
        dual_schedule_sampling=dual_schedule_sampling,
        # rescale_timesteps=rescale_timesteps,
    )
