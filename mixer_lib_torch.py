import einops
import flax
import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    params = []
    for j, (layer_sizes_, scale_) in enumerate(zip(layer_sizes, scale)):
        print(layer_sizes_, scale_)
        if len(layer_sizes_) == 2:
            bias = np.zeros([layer_sizes_[-1]])
        elif len(layer_sizes_) == 3:
            bias = np.zeros([layer_sizes_[0], layer_sizes_[2]])
        elif len(layer_sizes_) == 4:
            if layer_sizes_[-2] == 1:
                # Convolution
                bias = np.zeros([layer_sizes_[-1]])
            else:
                bias = np.zeros([layer_sizes_[0], layer_sizes_[1], layer_sizes_[3]])
        params.append((scale_ * rng.randn(*layer_sizes_), bias))
    return params


def get_param_scale(init_scheme, layer_sizes):
    if init_scheme == "kaiming":
        param_scale = [min(2.0 / math.sqrt(l[-2]), 0.1) for l in layer_sizes]
    elif init_scheme == "lecun":
        param_scale = [min(1.0 / math.sqrt(l[-2]), 0.1) for l in layer_sizes]
    elif init_scheme == "constant":
        param_scale = [0.1 for l in layer_sizes]
    else:
        assert False
    # Make last layer zero-init (read out).
    param_scale[-1] = 0.0
    return param_scale


def get_layer_sizes(
    metadata,
    num_patches,
    num_channel_mlp_units,
    num_blocks,
    num_groups,
    concat_groups,
    same_head,
    conv,
    ksize,
    num_proj_units=0,
    num_channel_mlp_hidden_units=-1,
    downsample=None,
    channel_ratio=None,
    group_ratio=None,
):
    num_classes = metadata["num_classes"]
    input_dim = (
        metadata["input_height"]
        * metadata["input_width"]
        // (num_patches**2)
        * metadata["input_channel"]
    )
    num_tokens = num_patches**2
    if same_head:
        assert concat_groups, "Same head only works with concat groups"
    layer_sizes = []
    loss_layer_sizes = []
    if num_channel_mlp_hidden_units < 0:
        num_channel_mlp_hidden_units = num_channel_mlp_units
    if channel_ratio is None:
        channel_ratio = [1] * num_blocks
    if group_ratio is None:
        group_ratio = [1] * num_blocks
    if downsample is None:
        downsample = [1] * num_blocks

    num_tokens_ = num_tokens
    num_channel_mlp_units_ = num_channel_mlp_units
    num_channel_mlp_hidden_units_ = num_channel_mlp_hidden_units
    num_groups_ = num_groups
    for blk in range(num_blocks):
        # Token mixing
        if blk > 0:
            if conv:
                layer_sizes.append([ksize, ksize, 1, num_channel_mlp_units_])
            else:
                layer_sizes.extend([[num_tokens_, num_tokens_]])

        # Channel mixing (group)
        if blk == 0:
            num_inp = input_dim
        else:
            num_inp = num_channel_mlp_units_ // channel_ratio[blk - 1]

        num_hid = num_channel_mlp_hidden_units_
        num_out = num_channel_mlp_units_

        layer_sizes.extend(
            [
                [num_inp, num_hid],
                [num_groups_, num_hid // num_groups_, num_out // num_groups_],
            ]
        )

        assert same_head

        if num_proj_units > 0:
            num_out_units = num_proj_units
        else:
            num_out_units = num_classes

        loss_layer_sizes.append([num_out, num_out_units])

        if downsample is not None:
            num_tokens_ = num_tokens_ // (downsample[blk] ** 2)
        if channel_ratio is not None:
            num_channel_mlp_units_ = num_channel_mlp_units_ * channel_ratio[blk]
            num_channel_mlp_hidden_units_ = (
                num_channel_mlp_hidden_units_ * channel_ratio[blk]
            )
        if group_ratio is not None:
            num_groups_ = num_groups_ * group_ratio[blk]

    if num_proj_units > 0:
        loss_layer_sizes.append([num_channel_mlp_units_, num_proj_units])
    loss_layer_sizes.append([num_channel_mlp_units_, num_classes])
    layer_sizes = layer_sizes + loss_layer_sizes
    return layer_sizes


# def layer_norm(x, gamma, beta, axis=-1, eps=1e-5):
#     mean = jnp.mean(x, axis=axis, keepdims=True)
#     mean_of_squares = jnp.mean(jnp.square(x), axis=axis, keepdims=True)
#     var = mean_of_squares - jnp.square(mean)
#     inv = jax.lax.rsqrt(var + eps)
#     if gamma is not None:
#         y = gamma * (x - mean) * inv
#     else:
#         y = (x - mean) * inv
#     if beta is not None:
#         y = y + beta
#     return y


def layer_norm(x, gamma, beta, axis=-1, eps=1e-5):
    layer_norm = nn.LayerNorm(x.size()[axis], eps=eps)
    y = layer_norm(x)
    if gamma is not None:
        y = gamma * y
    if beta is not None:
        y = y + beta
    return y


# def avg_pooling(inputs, stride=2, window=3):
#     B, P, D = inputs.shape[0], inputs.shape[1], inputs.shape[2:]
#     H = int(math.sqrt(P))
#     outputs = jnp.reshape(inputs, [B, H, H] + list(D))
#     outputs = flax.linen.avg_pool(
#         inputs, window_shape=(window, window), strides=(stride, stride), padding="SAME"
#     )
#     outputs = jnp.reshape(outputs, [B, -1] + list(D))
#     return outputs


def avg_pooling(inputs, stride=2, window=3):
    pool = nn.AvgPool2d(kernel_size=window, stride=stride, padding=1)
    return pool(inputs)


# def max_pooling(inputs, stride=2, window=3):
#     B, P, D = inputs.shape[0], inputs.shape[1], inputs.shape[2:]
#     H = int(math.sqrt(P))
#     outputs = jnp.reshape(inputs, [B, H, H] + list(D))
#     outputs = flax.linen.max_pool(
#         inputs, window_shape=(window, window), strides=(stride, stride), padding="SAME"
#     )
#     outputs = jnp.reshape(outputs, [B, -1] + list(D))
#     return outputs


def max_pooling(inputs, stride=2, window=3):
    pool = nn.MaxPool2d(kernel_size=window, stride=stride, padding=1)
    return pool(inputs)


def max_pooling(inputs, stride=2, window=3):
    B, P, *D = inputs.shape
    H = int(math.sqrt(P))
    outputs = inputs.view(B, H, H, *D)
    outputs = F.max_pool2d(
        outputs, kernel_size=window, stride=stride, padding=(window - 1) // 2
    )
    outputs = outputs.view(B, -1, *D)
    return outputs


# def fa_linear(inputs, fw_weight, fw_bias, bw_weight):
#     """Linear layer for feedback alignment."""
#     if len(inputs.shape) == 3:
#         if len(fw_weight.shape) == 3:
#             output = jnp.einsum("npc,pcd->npd", inputs, fw_weight)
#             bw_output = jnp.einsum("npc,pcd->npd", inputs, bw_weight)
#         else:
#             output = jnp.einsum("npc,cd->npd", inputs, fw_weight)
#             bw_output = jnp.einsum("npc,cd->npd", inputs, bw_weight)
#     else:
#         output = jnp.dot(inputs, fw_weight)
#         bw_output = jnp.dot(inputs, bw_weight)
#     return jax.lax.stop_gradient(output - bw_output) + bw_output + fw_bias


def fa_linear(inputs, fw_weight, fw_bias, bw_weight):
    """Linear layer for feedback alignment."""
    if len(inputs.shape) == 3:
        if len(fw_weight.shape) == 3:
            output = torch.einsum("npc,pcd->npd", inputs, fw_weight)
            bw_output = torch.einsum("npc,pcd->npd", inputs, bw_weight)
        else:
            output = torch.einsum("npc,cd->npd", inputs, fw_weight)
            bw_output = torch.einsum("npc,cd->npd", inputs, bw_weight)
    else:
        output = torch.matmul(inputs, fw_weight)
        bw_output = torch.matmul(inputs, bw_weight)
    return (output - bw_output).detach() + bw_output + fw_bias


# def linear(inputs, weight, bias=None):
#     if len(inputs.shape) == 3:
#         if len(weight.shape) == 3:
#             # No share weights.
#             output = jnp.einsum("npc,pcd->npd", inputs, weight) + bias
#         else:
#             output = jnp.einsum("npc,cd->npd", inputs, weight) + bias
#     else:
#         output = jnp.dot(inputs, weight)
#     if bias is not None:
#         output = output + bias
#     return output


def linear(inputs, weight, bias=None):
    if len(inputs.shape) == 3:
        if len(weight.shape) == 3:
            # No share weights.
            output = torch.einsum("npc,pcd->npd", inputs, weight)
        else:
            output = torch.einsum("npc,cd->npd", inputs, weight)
    else:
        output = torch.matmul(inputs, weight)

    if bias is not None:
        output = output + bias

    return output


# def fa_group_linear(inputs, fw_weight, fw_bias, bw_weight):
#     B, P, G, D = inputs.shape
#     if len(fw_weight.shape) == 4:
#         outputs = jnp.einsum("npgc,pgcd->npgd", inputs, fw_weight)
#         bw_outputs = jnp.einsum("npgc,pgcd->npgd", inputs, bw_weight)
#     elif len(fw_weight.shape) == 3:
#         outputs = jnp.einsum("npgc,gcd->npgd", inputs, fw_weight)
#         bw_outputs = jnp.einsum("npgc,gcd->npgd", inputs, bw_weight)
#     return jax.lax.stop_gradient(outputs - bw_outputs) + bw_outputs + fw_bias


def fa_group_linear(inputs, fw_weight, fw_bias, bw_weight):
    if len(inputs.shape) == 4:
        outputs = torch.einsum("npgc,pgcd->npgd", inputs, fw_weight)
        bw_outputs = torch.einsum("npgc,pgcd->npgd", inputs, bw_weight)
    elif len(inputs.shape) == 3:
        outputs = torch.einsum("npgc,gcd->npgd", inputs, fw_weight)
        bw_outputs = torch.einsum("npgc,gcd->npgd", inputs, bw_weight)
    return torch.add(outputs, bw_outputs, alpha=-1.0).detach() + bw_outputs + fw_bias


# def group_linear(inputs, weight, bias=None):
#     B, P, G, D = inputs.shape
#     if len(weight.shape) == 4:
#         outputs = jnp.einsum("npgc,pgcd->npgd", inputs, weight)
#     elif len(weight.shape) == 3:
#         # weight = jnp.reshape(weight, [G, weight.shape[0], -1])
#         outputs = jnp.einsum("npgc,gcd->npgd", inputs, weight)
#     if bias is not None:
#         outputs = outputs + bias
#     return outputs


def group_linear(inputs, weight, bias=None):
    outputs = torch.einsum("npgc,pgcd->npgd", inputs, weight)
    if bias is not None:
        outputs = outputs + bias
    return outputs


# def dropout_layer(x, key, drop, is_training=False):
#     if drop > 0.0:
#         if is_training:
#             key, subkey = jax.random.split(key)
#             keep = jax.random.bernoulli(subkey, 1.0 - drop, x.shape)
#             x = x * keep
#         else:
#             x = x * (1.0 - drop)
#     return x, key


def dropout_layer(x, drop, is_training=False):
    return F.dropout(x, p=drop, training=is_training)


# def depthwise_conv(x, weight):
#     return jax.lax.conv_general_dilated(
#         x,
#         weight,
#         window_strides=[1, 1],
#         dimension_numbers=("NHWC", "HWIO", "NHWC"),
#         padding="SAME",
#         feature_group_count=x.shape[-1],
#     )


def depthwise_conv(x, weight):
    return F.conv2d(x, weight, groups=x.size(1), padding=1)


# def normalize(x, swap=False, batch_norm=False, layer_norm_all=False):
#     if batch_norm:
#         outputs = layer_norm(x, None, None, axis=0)
#     elif layer_norm_all:
#         if len(x.shape) == 3:
#             outputs = layer_norm(x, None, None, axis=[1, 2])
#         elif len(x.shape) == 4:
#             outputs = layer_norm(x, None, None, axis=[1, 2, 3])
#     else:
#         if swap:
#             outputs = layer_norm(x, None, None, axis=1)
#         else:
#             outputs = layer_norm(x, None, None, axis=-1)
#     return outputs


def normalize(x, swap=False, batch_norm=False, layer_norm_all=False):
    if batch_norm:
        norm = nn.BatchNorm2d(x.size(0))
        outputs = norm(x)
    elif layer_norm_all:
        norm = nn.LayerNorm(x.size()[1:])
        outputs = norm(x)
    else:
        if swap:
            norm = nn.LayerNorm(x.size(1))
        else:
            norm = nn.LayerNorm(x.size(-1))
        outputs = norm(x)
    return outputs


# def normalize_images(images, mean_rgb, stddev_rgb):
#     """Normalize the image using ImageNet statistics."""
#     normed_images = images - jnp.array(mean_rgb).reshape((1, 1, 1, 3))
#     normed_images = normed_images / jnp.array(stddev_rgb).reshape((1, 1, 1, 3))
#     return normed_images


def normalize_images(images, mean_rgb, stddev_rgb):
    mean = torch.tensor(mean_rgb, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(stddev_rgb, dtype=torch.float32).view(1, -1, 1, 1)
    normed_images = (images - mean) / std
    return normed_images


# def preprocess(view, image_mean, image_std, num_patches):
#     #   num_patches = FLAGS.num_patches
#     patch_size = view.shape[1] // num_patches
#     view = normalize_images(view, image_mean, image_std)
#     view = jnp.reshape(
#         view,
#         [
#             view.shape[0],
#             num_patches,
#             patch_size,
#             num_patches,
#             patch_size,
#             view.shape[3],
#         ],
#     )
#     view = einops.rearrange(view, "n p h q w c -> n (p q) (h w c)")
#     return view


def preprocess(view, image_mean, image_std, num_patches):
    patch_size = view.shape[1] // num_patches
    view = normalize_images(view, image_mean, image_std)
    view = view.reshape(
        view.shape[0],
        num_patches,
        patch_size,
        num_patches,
        patch_size,
        view.shape[3],
    )
    view = view.permute(0, 1, 3, 2, 4, 5).contiguous()
    view = view.reshape(view.shape[0], num_patches * num_patches, -1)
    return view


NFIRST = 2
NLAYER = 3


def get_num_layers(blk):
    return NFIRST + NLAYER * (blk - 1)


def get_blk(i):
    if i < NFIRST:
        return 0, i
    else:
        return (i - NFIRST) // NLAYER + 1, (i - NFIRST) % NLAYER


def get_blk_idx(idx):
    if idx == 0:
        return 0, NFIRST
    else:
        return get_num_layers(idx), get_num_layers(idx + 1)


def get_blk_params(params, num_blocks, blk):
    NL = get_num_layers(num_blocks)
    if blk < num_blocks:
        start, end = get_blk_idx(blk)
        return params[start:end] + params[NL + blk : NL + blk + 1]
    else:
        return params[-1:]


def set_blk_params(params, num_blocks, blk, blk_params):
    NL = get_num_layers(num_blocks)
    start, end = get_blk_idx(blk)
    for q, p in enumerate(range(start, end)):
        params[p] = blk_params[q]
    # Last projection layer.
    params[NL + blk] = blk_params[-1]
    return params


def get_dataset_metadata(dataset):
    # MNIST_MEAN = (0.1307,)
    # MNIST_STD = (0.3081,)
    if dataset == "cifar-10":
        return {
            "num_classes": 10,
            "num_examples_train": 50000,
            "num_examples_test": 10000,
            "image_mean": (0.4914, 0.4822, 0.4465),
            "image_std": (0.2023, 0.1994, 0.2010),
            "input_height": 32,
            "input_width": 32,
            "input_channel": 3,
        }
    elif dataset == "imagenet-100":
        return {
            "num_classes": 100,
            "num_examples_train": 130000,
            "num_examples_test": 5000,
            "image_mean": (0.485, 0.456, 0.406),
            "image_std": (0.229, 0.224, 0.225),
            "input_height": 224,
            "input_width": 224,
            "input_channel": 3,
        }
    elif dataset == "imagenet2012":
        return {
            "num_classes": 1000,
            "num_examples_train": 1281167,
            "num_examples_test": 50000,
            "image_mean": (0.485, 0.456, 0.406),
            "image_std": (0.229, 0.224, 0.225),
            "input_height": 224,
            "input_width": 224,
            "input_channel": 3,
        }
    else:
        assert False
