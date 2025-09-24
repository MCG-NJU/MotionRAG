from torch.nn import Conv2d, Conv1d, Linear
from functools import reduce
from diffusers.models.lora import LoRALinearLayer, LoRAConv2dLayer
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear



LORA_FACTORY = {
        Conv2d: (lambda x, rank: LoRAConv2dLayer(
                     x.in_channels,
                     x.out_channels,
                     rank,
                 ),
                 lambda x, lora_x: LoRACompatibleConv(
                     x.in_channels,
                     x.out_channels,
                     x.kernel_size,
                     x.stride,
                     x.padding,
                     x.dilation,
                     x.groups,
                     x.bias is not None,
                     x.padding_mode,
                     lora_layer=lora_x
                 )
        ),
        LoRACompatibleConv:
            (lambda x, rank: LoRAConv2dLayer(
                x.in_channels,
                x.out_channels,
                rank
            ),
            lambda x: x),
        Linear:
            (lambda x, rank: LoRALinearLayer(
                x.in_features,
                x.out_features,
                rank
            ),
            lambda x, lora_x: LoRACompatibleLinear(
                 x.in_features,
                 x.out_features,
                 x.bias is not None,
                 lora_layer=lora_x
            ),
        ),
        LoRACompatibleLinear:
            (lambda x, rank: LoRALinearLayer(
                x.in_features,
                x.out_channels,
                rank,
                ),
            lambda x: x),
    }


def insert_lora_module(model, name, rank=64):

    split_name = name.split(".")
    parent_module = reduce(getattr, split_name[:-1], model)
    self_module = getattr(parent_module, split_name[-1])
    # lora_type, rmod_fn = LORA_FACTORY[type(self_module)]

    lora_layer = None
    for type, (lora_fn, rmod_fn) in LORA_FACTORY.items():
        if isinstance(self_module, type):
            lora_layer = lora_fn(self_module, rank)
            lora_layer.requires_grad_(True)

            rmod_layer = rmod_fn(self_module, None)
            rmod_layer.load_state_dict(self_module.state_dict())
            rmod_layer.requires_grad_(False)
            del self_module

            setattr(parent_module, split_name[-1], rmod_layer)
            setattr(rmod_layer, "lora_layer", lora_layer)
            break
    return lora_layer