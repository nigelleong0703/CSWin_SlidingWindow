from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# from cswin import CSWinTransformer
from timm.models.registry import register_model

# import CSwin
from CSwin import CSwin


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 100,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "cswin_224": _cfg(),
    "cswin_384": _cfg(crop_pct=1.0),
}

@register_model
def CSWin_64_12211_tiny_224_norm(pretrained=False, **kwargs):
    model = CSwin(
        mode="norm",
        patch_size=4,
        embed_dim=64,
        depth=[1, 2, 21, 1],
        split_size=[1, 2, 7, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        **kwargs
    )
    model.default_cfg = default_cfgs["cswin_224"]
    return model


@register_model
def CSWin_64_12211_tiny_224_sw1(pretrained=False, **kwargs):
    # model = CSwin(
    #     patch_size=4,
    #     embed_dim=64,
    #     depth=[1, 2, 21, 1],
    #     split_size=[1, 2, 7, 7],
    #     num_heads=[2, 4, 8, 16],
    #     mlp_ratio=4.0,
    #     **kwargs
    # )
    model = CSwin(
        mode="sw1",
        patch_size=4,
        embed_dim=64,
        depth=[1, 2, 21, 1],
        split_size=[1, 2, 7, 7],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.0,
        **kwargs
    )
    model.default_cfg = default_cfgs["cswin_224"]
    return model


@register_model
def CSWin_64_12211_tiny_224_sw2(pretrained=False, **kwargs):
    # model = CSwin(
    #     patch_size=4,
    #     embed_dim=64,
    #     depth=[1, 2, 21, 1],
    #     split_size=[1, 2, 7, 7],
    #     num_heads=[2, 4, 8, 16],
    #     mlp_ratio=4.0,
    #     **kwargs
    # )
    model = CSwin(
        mode="sw2",
        patch_size=4,
        embed_dim=64,
        depth=[1, 2, 21, 1],
        split_size=[1, 2, 7, 7],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.0,
        **kwargs
    )
    model.default_cfg = default_cfgs["cswin_224"]
    return model
