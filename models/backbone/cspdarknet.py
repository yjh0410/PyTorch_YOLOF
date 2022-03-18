"""
    This is a CSPDarkNet-53 with Mish.
"""
import torch
import torch.nn as nn


model_urls = {
    "cspdarknet53": "https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/cspdarknet53.pth",
}


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def ConvNormActivation(inplanes,
                       planes,
                       kernel_size=3,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1,
                       norm_type='BN'):
    """
    A help function to build a 'conv-bn-activation' module
    """
    layers = []
    layers.append(nn.Conv2d(inplanes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=False))
    if norm_type == 'BN':
        layers.append(nn.BatchNorm2d(planes, eps=1e-4, momentum=0.03))
    elif norm_type == 'FrozeBN':
        layers.append(FrozenBatchNorm2d(planes, eps=1e-4))
    layers.append(nn.Mish(inplace=True))
    return nn.Sequential(*layers)


def make_cspdark_layer(block,
                       inplanes,
                       planes,
                       num_blocks,
                       is_csp_first_stage,
                       dilation=1,
                       norm_type='BN'):
    downsample = ConvNormActivation(
        inplanes=planes,
        planes=planes if is_csp_first_stage else inplanes,
        kernel_size=1,
        stride=1,
        padding=0,
        norm_type=norm_type
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes if is_csp_first_stage else inplanes,
                downsample=downsample if i == 0 else None,
                dilation=dilation,
                norm_type=norm_type
            )
        )
    return nn.Sequential(*layers)


class DarkBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 dilation=1,
                 downsample=None,
                 norm_type='BN'):
        """Residual Block for DarkNet.
        This module has the dowsample layer (optional),
        1x1 conv layer and 3x3 conv layer.
        """
        super(DarkBlock, self).__init__()

        self.downsample = downsample

        if norm_type == 'BN':
            self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-4, momentum=0.03)
            self.bn2 = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.03)
        elif norm_type == 'FrozeBN':
            self.bn1 = FrozenBatchNorm2d(inplanes, eps=1e-4)
            self.bn2 = FrozenBatchNorm2d(planes, eps=1e-4)

        self.conv1 = nn.Conv2d(
            planes,
            inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.activation = nn.Mish(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out += identity

        return out


class CrossStagePartialBlock(nn.Module):
    """CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
    Refer to the paper for more details: https://arxiv.org/abs/1911.11929.
    In this module, the inputs go throuth the base conv layer at the first,
    and then pass the two partial transition layers.
    1. go throuth basic block (like DarkBlock)
        and one partial transition layer.
    2. go throuth the other partial transition layer.
    At last, They are concat into fuse transition layer.
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels
        stage_layers (nn.Module): the basic block which applying CSPNet.
        is_csp_first_stage (bool): Is the first stage or not.
            The number of input and output channels in the first stage of
            CSPNet is different from other stages.
        dilation (int): conv dilation
        stride (int): stride for the base layer
    """

    def __init__(self,
                 inplanes,
                 planes,
                 stage_layers,
                 is_csp_first_stage,
                 dilation=1,
                 stride=2,
                 norm_type='BN'):
        super(CrossStagePartialBlock, self).__init__()

        self.base_layer = ConvNormActivation(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_type=norm_type
        )
        self.partial_transition1 = ConvNormActivation(
            inplanes=planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type
        )
        self.stage_layers = stage_layers

        self.partial_transition2 = ConvNormActivation(
            inplanes=inplanes if not is_csp_first_stage else planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type
        )
        self.fuse_transition = ConvNormActivation(
            inplanes=planes if not is_csp_first_stage else planes * 2,
            planes=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type
        )

    def forward(self, x):
        x = self.base_layer(x)

        out1 = self.partial_transition1(x)

        out2 = self.stage_layers(x)
        out2 = self.partial_transition2(out2)

        out = torch.cat([out2, out1], dim=1)
        out = self.fuse_transition(out)

        return out


class CSPDarkNet53(nn.Module):
    """CSPDarkNet backbone.
    Refer to the paper for more details: https://arxiv.org/pdf/1804.02767
    Args:
        depth (int): Depth of Darknet, from {53}.
        num_stages (int): Darknet stages, normally 5.
        with_csp (bool): Use cross stage partial connection or not.
        out_features (List[str]): Output features.
        norm_type (str): type of normalization layer.
        res5_dilation (int): dilation for the last stage
    """

    def __init__(self, norm_type='BN', res5_dilation=1):
        super(CSPDarkNet53, self).__init__()

        self.block =  DarkBlock
        self.stage_blocks = (1, 2, 8, 8, 4)
        self.with_csp = True
        self.inplanes = 32
        self.res5_dilation = res5_dilation

        self.backbone = nn.ModuleDict()
        self.layer_names = []
        print(norm_type)
        # First stem layer
        self.backbone["conv1"] = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        if norm_type == 'BN':
            self.backbone["bn1"] = nn.BatchNorm2d(self.inplanes, eps=1e-4, momentum=0.03)
        elif norm_type == 'FrozeBN':
            self.backbone["bn1"] = FrozenBatchNorm2d(self.inplanes, eps=1e-4)
        self.backbone["act1"] = nn.Mish(inplace=True)

        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i
            dilation = 1
            stride = 2
            if i == 4 and self.res5_dilation == 2:
                dilation = self.res5_dilation
                stride = 1
            layer = make_cspdark_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                is_csp_first_stage=True if i == 0 else False,
                dilation=dilation,
                norm_type=norm_type
            )
            layer = CrossStagePartialBlock(
                self.inplanes,
                planes,
                stage_layers=layer,
                is_csp_first_stage=True if i == 0 else False,
                dilation=dilation,
                stride=stride,
                norm_type=norm_type
            )
            self.inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            self.backbone[layer_name]=layer
            self.layer_names.append(layer_name)


    def freeze(self):
        # freeze stem
        print('freeze stem ...')
        for p in self.backbone["conv1"].parameters():
            p.requires_grad = False
        # freeze stage-1
        print('freeze stage-1 ...')
        for p in self.backbone['layer1'].parameters():
            p.requires_grad = False


    def forward(self, x):
        output = []
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["act1"](x)

        for i, layer_name in enumerate(self.layer_names):
            layer = self.backbone[layer_name]
            x = layer(x)

        return x


def cspdarknet53(pretrained=False, res5_dilation=1, norm_type='BN'):
    """
    Create a CSPDarkNet.
    """
    model = CSPDarkNet53(norm_type=norm_type, res5_dilation=res5_dilation)
    # load weight
    if pretrained:
        print('Loading pretrained cspdarknet53 ...')
        url = model_urls['cspdarknet53']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)

    # freeze stem and stage-1
    model.freeze()

    return model


def build_cspdarknet(model_name='cspdarknet53', pretrained=False, norm_type='BN'):
    if model_name == 'cspdarknet53':
        backbone = cspdarknet53(pretrained=pretrained, 
                                norm_type=norm_type)
        feat_dim = 1024

    elif model_name == 'cspdarknet53-d':
        backbone = cspdarknet53(pretrained=pretrained, 
                                res5_dilation=2,
                                norm_type=norm_type)
        feat_dim = 1024

    else:
        print('Unknown Version of CSPDarkNet53 !!')
        exit()

    return backbone, feat_dim


if __name__=='__main__':
    img_size = 640
    input = torch.ones(1, 3, img_size, img_size)

    model, feat_dim = build_cspdarknet('cspdarknet53-d', pretrained=True)
    y = model(input)
    print(y.size())