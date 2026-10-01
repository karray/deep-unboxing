"""Layer-wise relevance propagation (LRP) that starts from the features of a vision model.

Classic LRP explains one output score. Here there is no score, so we start
from the spatial features themselves: a total amount of 1, spread over the
feature values (channel x position), either evenly or in proportion to how
active each value is. LRP then passes this amount back through the network,
layer by layer, until it reaches the pixels.

How the amount is split at each layer is decided by *rules* (zennit provides
them). Modern vision models contain operations that zennit does not handle out of
the box, so this module adds three things:

1. Canonizers rewrite a few forward passes so that every residual "+" becomes
   an explicit ``Sum`` module (a place where a rule can decide how to split),
   attention weights are held constant, and the LayerNorm denominator is
   treated as a constant.
2. A composite assigns a rule to every layer by its role (stem, block,
   downsampling, normalization, residual sum).
3. ``explain`` runs the forward pass, puts the starting amount on the
   features and collects the relevance at the input pixels with autograd.

This is a condensed version of the rules used for the blog post. They are one
reasonable set of choices, not the only one.
"""
from collections import Counter
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from torch.overrides import TorchFunctionMode
from timm.layers import LayerNorm2d, LayerScale
from timm.layers.activations import GELU as TimmGELU, GELUTanh
from timm.layers.attention import Attention
from timm.layers.squeeze_excite import SEModule
from timm.models import convnext, eva, maxxvit, resnet, swin_transformer, vision_transformer
from zennit.canonizers import AttributeCanonizer
from zennit.composites import EpsilonPlus, LayerMapComposite
from zennit.core import Hook
from zennit.layer import Sum
from zennit.rules import Epsilon, Gamma, Norm, Pass, WSquare, ZPlus
from zennit.torchvision import ResNetBottleneckCanonizer
from zennit.types import Activation, AvgPool, BatchNorm


# ============================================================================
# 1. Small helper modules, so that a rule can be attached to them
# ============================================================================

class LayerScaleConst(nn.Module):
    """x * gamma with gamma treated as a constant (ConvNeXt and Eva layer scale)."""

    def __init__(self, gamma, channels_first=False):
        super().__init__()
        self.gamma = gamma.detach()   # a plain tensor, so it is not registered as a second parameter
        self.channels_first = channels_first

    def forward(self, x):
        return x * (self.gamma.reshape(1, -1, 1, 1) if self.channels_first else self.gamma)


class AttentionMix(nn.Module):
    """Values mixed by attention weights that are held constant (set just before the call)."""

    def forward(self, v):
        return self.weights @ v


class GatedSignal(nn.Module):
    """x * gate with the gate held constant (squeeze-and-excitation)."""

    def forward(self, x):
        return x * self.gate


def _mix(attn, q, k, v, bias=None):
    """Attention with its weights computed from detached queries and keys, so they act as constants."""
    logits = (q.detach() * attn.scale) @ k.detach().transpose(-2, -1)
    if bias is not None:
        logits = logits.masked_fill(~bias, float('-inf')) if bias.dtype == torch.bool else logits + bias
    attn.lrp_mix.weights = logits.softmax(dim=-1)
    return attn.lrp_mix(v)


def _add(sum_module, a, b):
    """a + b through an explicit Sum module, so the residual split gets a rule."""
    return sum_module(torch.stack((a, b), dim=-1))


# ============================================================================
# 2. Canonizers: forward passes rewritten with explicit sums and fixed attention
#    (written for timm 1.0.x; the math is unchanged)
# ============================================================================

class ExplicitResiduals(AttributeCanonizer):
    """Explicit residual sums for timm ResNet, ConvNeXt, ViT, Eva, Swin and MaxViT blocks."""

    def __init__(self):
        super().__init__(self.attributes)

    @classmethod
    def attributes(cls, name, module):
        if isinstance(module, resnet.Bottleneck):
            return {'forward': cls.resnet_bottleneck.__get__(module), 'lrp_sum1': Sum(dim=-1)}
        if isinstance(module, convnext.ConvNeXtBlock):
            extra = {'lrp_scale': LayerScaleConst(module.gamma, channels_first=True)} if module.gamma is not None else {}
            return {'forward': cls.convnext_block.__get__(module), 'lrp_sum1': Sum(dim=-1), **extra}
        if isinstance(module, vision_transformer.Block):
            return {'forward': cls.vit_block.__get__(module), 'lrp_sum1': Sum(dim=-1), 'lrp_sum2': Sum(dim=-1)}
        if isinstance(module, eva.EvaBlock):
            extra = {} if module.gamma_1 is None else {'lrp_scale1': LayerScaleConst(module.gamma_1),
                                                       'lrp_scale2': LayerScaleConst(module.gamma_2)}
            return {'forward': cls.eva_block.__get__(module), 'lrp_sum1': Sum(dim=-1), 'lrp_sum2': Sum(dim=-1), **extra}
        if isinstance(module, swin_transformer.SwinTransformerBlock):
            return {'forward': cls.swin_block.__get__(module), 'lrp_sum1': Sum(dim=-1), 'lrp_sum2': Sum(dim=-1)}
        if isinstance(module, maxxvit.PartitionAttentionCl):
            return {'forward': cls.partition_block.__get__(module), 'lrp_sum1': Sum(dim=-1), 'lrp_sum2': Sum(dim=-1)}
        if isinstance(module, maxxvit.MbConvBlock):
            return {'forward': cls.mbconv_block.__get__(module), 'lrp_sum1': Sum(dim=-1)}
        if isinstance(module, SEModule):
            return {'forward': cls.squeeze_excite.__get__(module), 'lrp_gated': GatedSignal()}
        return None

    @staticmethod
    def resnet_bottleneck(self, x):
        shortcut = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.aa(self.act2(self.drop_block(self.bn2(self.conv2(x)))))
        x = self.bn3(self.conv3(x))
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.act3(_add(self.lrp_sum1, x, shortcut))

    @staticmethod
    def convnext_block(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.mlp(self.norm(x))
        else:
            x = self.mlp(self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = self.lrp_scale(x)
        return _add(self.lrp_sum1, self.drop_path(x), self.shortcut(shortcut))

    @staticmethod
    def vit_block(self, x, attn_mask=None, is_causal=False):
        branch = self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = _add(self.lrp_sum1, x, branch)
        branch = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return _add(self.lrp_sum2, x, branch)

    @staticmethod
    def eva_block(self, x, rope=None, attn_mask=None, is_causal=False):
        branch = self.attn(self.norm1(x), rope=rope)
        if self.gamma_1 is not None:
            branch = self.lrp_scale1(branch)
        x = _add(self.lrp_sum1, x, self.drop_path1(branch))
        branch = self.mlp(self.norm2(x))
        if self.gamma_2 is not None:
            branch = self.lrp_scale2(branch)
        return _add(self.lrp_sum2, x, self.drop_path2(branch))

    @staticmethod
    def swin_block(self, x):
        B, H, W, C = x.shape
        branch = self.drop_path1(self._attn(self.norm1(x)))
        x = _add(self.lrp_sum1, x, branch).reshape(B, -1, C)
        branch = self.drop_path2(self.mlp(self.norm2(x)))
        return _add(self.lrp_sum2, x, branch).reshape(B, H, W, C)

    @staticmethod
    def partition_block(self, x):
        branch = self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = _add(self.lrp_sum1, x, branch)
        branch = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return _add(self.lrp_sum2, x, branch)

    @staticmethod
    def mbconv_block(self, x):
        shortcut = self.shortcut(x)
        x = self.norm1(self.conv1_1x1(self.down(self.pre_norm(x))))
        x = self.conv2_kxk(x)
        if self.se_early is not None:
            x = self.se_early(x)
        x = self.norm2(x)
        if self.se is not None:
            x = self.se(x)
        return _add(self.lrp_sum1, self.drop_path(self.conv3_1x1(x)), shortcut)

    @staticmethod
    def squeeze_excite(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc2(self.act(self.bn(self.fc1(x_se))))
        self.lrp_gated.gate = self.gate(x_se).detach()
        return self.lrp_gated(x)


class FixedAttention(AttributeCanonizer):
    """Attention whose weights are held constant, so relevance flows only through the values."""

    def __init__(self):
        super().__init__(self.attributes)

    @classmethod
    def attributes(cls, name, module):
        forwards = {Attention: cls.vit, eva.EvaAttention: cls.eva,
                    swin_transformer.WindowAttention: cls.window, maxxvit.AttentionCl: cls.attention_cl}
        forward = forwards.get(type(module))
        return None if forward is None else {'forward': forward.__get__(module), 'lrp_mix': AttentionMix()}

    @staticmethod
    def vit(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = _mix(self, self.q_norm(q), self.k_norm(k), v).transpose(1, 2).reshape(B, N, self.attn_dim)
        return self.proj_drop(self.proj(self.norm(x)))

    @staticmethod
    def eva(self, x, rope=None, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                qkv = self.qkv(x) + qkv_bias if self.qkv_bias_separate else \
                    F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            q, k, v = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        else:
            q, k, v = (proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
                       for proj in (self.q_proj, self.k_proj, self.v_proj))
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:  # rotary position embedding on the patch tokens only
            npt, half = self.num_prefix_tokens, getattr(self, 'rotate_half', False)
            q = torch.cat([q[:, :, :npt], eva.apply_rot_embed_cat(q[:, :, npt:], rope, half=half)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt], eva.apply_rot_embed_cat(k[:, :, npt:], rope, half=half)], dim=2).type_as(v)
        x = _mix(self, q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(self.norm(x)))

    @staticmethod
    def window(self, x, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        bias = self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            bias = (bias.expand(B_, -1, -1, -1).reshape(-1, num_win, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0)).reshape(-1, self.num_heads, N, N)
        return self.proj_drop(self.proj(_mix(self, q, k, v, bias).transpose(1, 2).reshape(B_, N, -1)))

    @staticmethod
    def attention_cl(self, x, shared_rel_pos=None):
        B, restore_shape = x.shape[0], x.shape[:-1]
        if self.head_first:
            q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        else:
            q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)
        bias = self.rel_pos.get_bias() if self.rel_pos is not None else shared_rel_pos
        x = _mix(self, q, k, v, bias).transpose(1, 2).reshape(restore_shape + (-1,))
        return self.proj_drop(self.proj(x))


class ConstantLayerNormScale(AttributeCanonizer):
    """LayerNorm with its denominator (the standard deviation) treated as a constant.

    The forward result is unchanged; only the backward pass sees a linear map.
    """

    def __init__(self):
        super().__init__(self.attributes)

    @classmethod
    def attributes(cls, name, module):
        if isinstance(module, LayerNorm2d):
            return {'forward': cls.channels_first.__get__(module)}
        if isinstance(module, nn.LayerNorm):
            return {'forward': cls.channels_last.__get__(module)}
        return None

    @staticmethod
    def channels_last(self, x):
        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        centered = x - x.mean(dim=dims, keepdim=True)
        y = centered / torch.sqrt(centered.square().mean(dim=dims, keepdim=True) + self.eps).detach()
        if self.elementwise_affine:
            y = y * self.weight
            if self.bias is not None:
                y = y + self.bias
        return y

    @staticmethod
    def channels_first(self, x):
        centered = x - x.mean(dim=1, keepdim=True)
        y = centered / torch.sqrt(centered.square().mean(dim=1, keepdim=True) + self.eps).detach()
        if self.elementwise_affine:
            y = y * self.weight.reshape(1, -1, 1, 1)
            if self.bias is not None:
                y = y + self.bias.reshape(1, -1, 1, 1)
        return y


# ============================================================================
# 3. Rules for the residual sums and the layer-role composite
# ============================================================================

class EqualSplit(Hook):
    """Give each residual branch the same share of the relevance."""

    def backward(self, module, grad_input, grad_output):
        shape = grad_input[0].shape
        return (grad_output[0].unsqueeze(module.dim).expand(shape) / shape[module.dim],)


class MagnitudeSplit(Hook):
    """Split the relevance in proportion to each branch's magnitude |contribution|."""

    def forward(self, module, args, kwargs, output):
        self.stored_tensors['branches'] = args[0].detach()

    def backward(self, module, grad_input, grad_output):
        weights = self.stored_tensors['branches'].abs()
        peak = weights.amax(dim=module.dim, keepdim=True)
        weights = weights / torch.where(peak > 0, peak, torch.ones_like(peak))
        denom = weights.sum(dim=module.dim, keepdim=True)
        fraction = torch.where(denom > 0, weights / denom.clamp_min(1),
                               torch.full_like(weights, 1 / weights.shape[module.dim]))
        return (fraction * grad_output[0].unsqueeze(module.dim),)


class RoleComposite(LayerMapComposite):
    """Rules by layer role.

    stem (first convolution): WSquare or z+; mixing layers inside blocks (linear,
    depthwise and 1x1 convolutions): Gamma(0.25); other convolutions (downsampling):
    z+; normalization and attention mixing: Epsilon; residual sums and pooling:
    Norm (split by contribution); activations, layer scale and gates: Pass.
    """

    def __init__(self, canonizers, stem_rule, epsilon=1e-4, stabilizer=1e-6):
        super().__init__(layer_map=[
            (nn.LayerNorm, Epsilon(epsilon=epsilon)),
            ((TimmGELU, GELUTanh, Activation, LayerScale, LayerScaleConst, GatedSignal), Pass()),
            (AttentionMix, Epsilon(epsilon=epsilon)),
            ((Sum, AvgPool), Norm(stabilizer=stabilizer)),
        ], canonizers=canonizers)
        self.block, self.stem, self.resampling = Gamma(gamma=0.25), stem_rule, ZPlus(stabilizer=stabilizer)
        self.merged_batch_norm, self.batch_norm = Pass(), Epsilon(epsilon=epsilon)

    def mapping(self, ctx, name, module):
        if isinstance(module, BatchNorm):  # a merged batch norm is the identity (eps set to 0)
            return self.merged_batch_norm if module.eps == 0 else self.batch_norm
        if isinstance(module, nn.Conv2d):
            if not ctx.get('stem'):
                ctx['stem'] = name
                return self.stem
            depthwise = module.groups > 1 and module.groups == module.in_channels
            return self.block if depthwise or module.kernel_size == (1, 1) else self.resampling
        if isinstance(module, nn.Linear):
            return self.block
        return super().mapping(ctx, name, module)


# The rules for each architecture family:
#   residual: how a residual sum splits relevance between its branches, 'equal',
#             'magnitude' or 'activation' (Norm: in proportion to the signed contributions)
#   stem:     the rule for the first convolution, 'wsquare' or 'zplus'
#   start:    how the starting amount is spread over the features, see ``explain``
LRP_SETTINGS = {
    'resnet':   {'residual': 'equal', 'stem': 'zplus', 'start': 'uniform'},
    'convnext': {'residual': 'activation', 'stem': 'wsquare', 'start': 'activation'},
    'swin':     {'residual': 'activation', 'stem': 'wsquare', 'start': 'uniform'},
    'maxvit':   {'residual': 'magnitude', 'stem': 'wsquare', 'start': 'uniform'},
    'vit':      {'residual': 'magnitude', 'stem': 'zplus', 'start': 'uniform'},
    'eva':      {'residual': 'magnitude', 'stem': 'zplus', 'start': 'uniform'},
}


def make_composite(family):
    """The rule set for one architecture family."""
    settings = LRP_SETTINGS[family]
    if family == 'resnet':
        composite = EpsilonPlus(canonizers=[ResNetBottleneckCanonizer(), ExplicitResiduals()])
    else:
        stem_rule = {'wsquare': WSquare, 'zplus': ZPlus}[settings['stem']]()
        composite = RoleComposite([ExplicitResiduals(), FixedAttention(), ConstantLayerNormScale()], stem_rule)
    if settings['residual'] != 'activation':
        split = {'equal': EqualSplit, 'magnitude': MagnitudeSplit}[settings['residual']]
        composite.layer_map.insert(0, (Sum, split()))
    return composite


# ============================================================================
# 4. Plumbing: fold batch norms into the preceding layer (only where that is exact)
# ============================================================================

class _CountUses(TorchFunctionMode):
    """Count how many operations consume each tensor during one forward pass."""
    # reading a tensor's shape or dtype is not a use
    _metadata = {getattr(torch.Tensor, n).__get__ for n in
                 ('shape', 'dtype', 'device', 'ndim', 'requires_grad', 'layout', 'is_cuda', 'grad_fn')} | \
                {torch.Tensor.size, torch.Tensor.dim, torch.Tensor.numel, torch.Tensor.stride, torch.Tensor.is_contiguous}

    def __init__(self):
        super().__init__()
        self.uses = Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in self._metadata:
            stack = list(args) + list(kwargs.values())
            while stack:
                value = stack.pop()
                if isinstance(value, torch.Tensor):
                    self.uses[id(value)] += 1
                elif isinstance(value, (list, tuple)):
                    stack.extend(value)
                elif isinstance(value, dict):
                    stack.extend(value.values())
        return func(*args, **kwargs)


@contextmanager
def merged_batch_norms(model, x):
    """Merge each batch norm into the conv/linear layer that feeds it and nothing else."""
    produced, pairs, handles, counter = {}, [], [], _CountUses()

    def remember_output(layer, inputs, output):   # which layer produced a tensor, and how often it was used by then
        produced[id(output)] = (layer, output, counter.uses[id(output)])

    def remember_input(bn, inputs):                # which tensor a batch norm reads
        pairs.append((bn, inputs[0]))

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(remember_output))
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            handles.append(module.register_forward_pre_hook(remember_input))
    try:
        with torch.no_grad(), counter:
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    merges = []
    for bn, source in pairs:
        layer, output, start = produced.get(id(source), (None, None, 0))
        if output is source and counter.uses[id(source)] - start == 1:
            merges.append((layer, bn))
    def frozen(t):
        return nn.Parameter(t.detach().clone(), requires_grad=False)

    saved = []
    try:
        for layer, bn in dict.fromkeys(merges):
            saved += [(layer, dict(layer._parameters), dict(layer._buffers), None),
                      (bn, dict(bn._parameters), dict(bn._buffers), bn.eps)]
            weight = bn.weight if bn.weight is not None else torch.ones_like(bn.running_var)
            shift = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
            scale = weight / (bn.running_var + bn.eps).sqrt()
            index = (slice(None),) + (None,) * (layer.weight.ndim - 1)
            bias = layer.bias if layer.bias is not None else torch.zeros_like(bn.running_mean)
            layer._parameters['weight'] = frozen(layer.weight * scale[index])
            layer._parameters['bias'] = frozen((bias - bn.running_mean) * scale + shift)
            for key, fill in (('weight', 1.), ('bias', 0.)):
                if bn._parameters.get(key) is not None:
                    bn._parameters[key] = frozen(torch.full_like(bn._parameters[key], fill))
            bn._buffers['running_mean'] = torch.zeros_like(bn.running_mean)
            bn._buffers['running_var'] = torch.ones_like(bn.running_var)
            bn.eps = 0.
        yield
    finally:
        for module, parameters, buffers, eps in reversed(saved):
            module._parameters.clear()
            module._parameters.update(parameters)
            module._buffers.clear()
            module._buffers.update(buffers)
            if eps is not None:
                module.eps = eps


# ============================================================================
# 5. Explain
# ============================================================================

def explain(model, x, start=None, exclude=()):
    """Relevance of every input pixel for the model's spatial features.

    model:   a ``xai_utils.models.FeatureExtractor`` (knows its family and rules).
    x:       a preprocessed 1x3xHxW tensor.
    start:   how the initial amount of 1 is spread over the feature values
             (channel x position). 'uniform' gives every value the same share.
             'activation' gives each value a share in proportion to its
             positive part, so that strongly active features start with more
             relevance (the positive part, because these features are signed
             and their raw sum can be negative). By default, the setting of
             the model's family in ``LRP_SETTINGS``.
    exclude: feature cells (row, column) that get no starting amount at all.
    Returns the relevance as a 3xHxW tensor (one map per colour channel).

    LRP runs on the CPU: it takes a few seconds per image and gives the same
    numbers on every machine (GPU kernels differ in the last digits).
    """
    device = next(model.parameters()).device
    model.cpu().eval().requires_grad_(False)
    x = x.detach().cpu().clone().requires_grad_(True)
    family = model.info.family
    composite = make_composite(family)
    try:
        with torch.enable_grad(), merged_batch_norms(model, x), composite.context(model):
            features = model(x)
            if (start or LRP_SETTINGS[family]['start']) == 'uniform':
                amount = torch.ones_like(features)
            else:
                amount = features.detach().clamp(min=0)
            for row, col in exclude:
                amount[..., row, col] = 0
            relevance, = torch.autograd.grad(features, x, amount / amount.sum())
    finally:
        model.to(device)
    return relevance[0].detach()
