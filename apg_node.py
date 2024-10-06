
import torch


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0
        self.prev_sigma = None

    def update(self, update_value: torch.Tensor, sigma: float):
        if self.prev_sigma is not None and sigma > self.prev_sigma:
            self.running_average = 0
        self.prev_sigma = sigma

        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(v0: torch.Tensor, v1: torch.Tensor):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(pred_cond: torch.Tensor,
                        pred_uncond: torch.Tensor, 
                        guidance_scale: float, 
                        sigma: float,
                        momentum_buffer: MomentumBuffer = None, 
                        eta: float = 1.0, 
                        norm_threshold: float = 0.0):
    
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff, sigma)
        diff = momentum_buffer.running_average
    
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    
    return pred_guided


class APGFunctionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "norm_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_momentum": (["enable", "disable"],),
                "momentum": ("FLOAT", {"default": 0.05, "min": -1.00, "max": 1.00, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "apg"

    def patch(self, model, eta, norm_threshold, use_momentum, momentum):
        m = model.clone()

        momentum_buffer = MomentumBuffer(momentum) if use_momentum == "enable" else None

        def cfg_function(args):
            sigma = args["sigma"]
            cond_pred = args["cond"]
            uncond_pred = args["uncond"]
            cfg = args["cond_scale"]

            if uncond_pred is None:
                return uncond_pred + (cond_pred - uncond_pred) * cfg

            return normalized_guidance(
                cond_pred,
                uncond_pred, 
                cfg,
                sigma.item(),
                momentum_buffer=momentum_buffer,
                eta=eta,
                norm_threshold=norm_threshold
            )

        m.set_model_sampler_cfg_function(cfg_function)

        return (m,)

