from .basic import *
#from versions.faster_rcnn.rpn import AnchorGenerator
#from versions.faster_rcnn.faster_rcnn import FasterRCNN
#from torchvision.models.detection import FasterRCNN
from  dect.model.self_vision.models.detection import  FasterRCNN
from dect.model.self_vision.ops.poolers import SingleScalePsRoIAlign
from dect.model.self_vision.models.detection.rpn import AnchorGenerator
from dect.config.config import my_configs
import collections

soupnet_version = my_configs.get("Snet_version")
PS_ROI_CHANNELS = my_configs.get("PS_ROI_CHANNELS")
PS_ROI_WINDOW_SIZE = my_configs.get("PS_ROI_WINDOW_SIZE")
anchor_numb = my_configs.get("anchor_number")
RPN_FEATURE_CHANNELS = my_configs.get("rpn_dense")
REPRESENT_CHANNELS = my_configs.get("representation_size")
CLASS_NUM = my_configs.get("num_classes")
anchor_sizes = my_configs.get("anchor_sizes")
aspect_ratios = my_configs.get("aspect_ratios")
Multi_size = my_configs.get("Multi_size")

SOUPNET_VERSION_FAST = 49

SOUPNET_VERSION_BALANCE = 146

SOUPNET_VERSION_ACCURATE = 535

CONTEXT_ENHANCE_CHANNELS = PS_ROI_CHANNELS * PS_ROI_WINDOW_SIZE * PS_ROI_WINDOW_SIZE


def get_model(version):

    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                       aspect_ratios=aspect_ratios)

    backbone = Soupnet(version)

    backbone = ContextEnhane(backbone)

    backbone.out_channels = CONTEXT_ENHANCE_CHANNELS

    rpn_head = RPN(CONTEXT_ENHANCE_CHANNELS, RPN_FEATURE_CHANNELS)

    spatial_attention_model = SpatialAttention(RPN_FEATURE_CHANNELS, CONTEXT_ENHANCE_CHANNELS)

    roi_pooler = SingleScalePsRoIAlign(featmap_names=['0'], output_size=PS_ROI_WINDOW_SIZE, sampling_ratio=-1)

    feature_merge_fc = FeatureMergeFc(PS_ROI_CHANNELS, REPRESENT_CHANNELS, feature_size=PS_ROI_WINDOW_SIZE)

    box_predictor = ClassAndBoxPredictor(REPRESENT_CHANNELS, CLASS_NUM, feature_size=1)

    model = FasterRCNN(backbone,
                       num_classes=None,
                       rpn_anchor_generator=anchor_generator,
                       #Multi_size=Multi_size,
                       box_roi_pool=roi_pooler,
                       rpn_head=rpn_head,
                       attention=spatial_attention_model,
                       # sam_model=spatial_attention_model,
                       box_head=feature_merge_fc,
                       box_predictor=box_predictor,
                       min_size=320, max_size=320
                       )

    return model

class Soupnet(nn.Module):
    cfg = {
        SOUPNET_VERSION_FAST: [24, 60, 120, 240, 512],
        SOUPNET_VERSION_BALANCE: [24, 132, 264, 528],
        SOUPNET_VERSION_ACCURATE: [48, 248, 496, 992],
    }

    def __init__(self,  version=SOUPNET_VERSION_FAST, **kwargs):
        super(Soupnet, self).__init__()

        self.version = version

        num_layers = [4, 8, 4]
        self.num_layers = num_layers
        channels = self.cfg[version]
        self.channels = channels

        self.conv1 = conv_relu(3, channels[0], kernel_size=3, stride=2, pad=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_layer(
            num_layers[0], channels[0], channels[1], **kwargs)

        self.stage2 = self._make_layer(
            num_layers[1], channels[1], channels[2], **kwargs)

        self.C4_channels = channels[2]

        self.stage3 = self._make_layer(
            num_layers[2], channels[2], channels[3], **kwargs)

        self.C5_channels = channels[3]
        self.Cglb_channels = channels[3]

        if (5 == len(self.channels)):
            self.conv5 = conv_bn(
                channels[3], channels[4], kernel_size=1, stride=1 ,pad=0)

            self.C5_channels = channels[4]
            self.Cglb_channels = channels[4]

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, num_layers, in_channels, out_channels, **kwargs):
        layers = [DownBlock(in_channels, out_channels, **kwargs)]
        for i in range(num_layers - 1):
            layers.append(BasicBlock(out_channels, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        if len(self.channels) == 5:
            c5 = self.conv5(c5)

        Cglb = self.avgpool(c5)

        return c4, c5, Cglb


class ContextEnhane(nn.Module):

    def  __init__(self, backone):
        super(ContextEnhane, self).__init__()

        self.backone = backone
        self.conv_c4 = conv(backone.C4_channels, CONTEXT_ENHANCE_CHANNELS, kernel_size=1, stride=1, pad=0)
        self.conv_c5 = conv(backone.C5_channels, CONTEXT_ENHANCE_CHANNELS, kernel_size=1, stride=1, pad=0)
        self.conv_glb = conv(backone.Cglb_channels, CONTEXT_ENHANCE_CHANNELS, kernel_size=1, stride=1, pad=0)
        self.unsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):

        [C4, C5, Cglb] = self.backone(x)
        C4_cem = self.conv_c4(C4)
        C5_cem = self.conv_c5(C5)
        C5_cem = self.unsample(C5_cem)
        Cglb_cem = self.conv_glb(Cglb)

        return C4_cem + C5_cem + Cglb_cem

class RPN(nn.Module):

    def __init__(self, context_enhance_channels, region_proposal_input_channels):
        super(RPN, self).__init__()
        self.anchor_num = anchor_numb

        self.dwconv5x5_bn = dwconv_bn(context_enhance_channels, context_enhance_channels,
                                         kernel_size=5, stride=1, pad=2)
        self.conv1x1_bn_relu = conv_bn_relu(context_enhance_channels, region_proposal_input_channels,
                                            kernel_size=1, stride=1, pad=0)

        self.class_predict_conv = conv(region_proposal_input_channels, self.anchor_num,
                                       kernel_size=1, stride=1, pad=0)

        self.location_predict_conv = conv(region_proposal_input_channels, 4 * self.anchor_num,
                                          kernel_size=1, stride=1, pad=0)

    def forward(self, input):
        logits = []
        bbox_reg = []
        rpn_feature_tensor_list = []

        for input_feature in input:
            out = self.dwconv5x5_bn(input_feature)
            out = self.conv1x1_bn_relu(out)

            logits.append(self.class_predict_conv(out))
            bbox_reg.append(self.location_predict_conv(out))
            rpn_feature_tensor_list.append(out)

        return logits, bbox_reg, rpn_feature_tensor_list

class SpatialAttention(torch.nn.Module):
    def __init__(self, class_predict_input_channels, context_enhance_channels):
        super(SpatialAttention, self).__init__()

        self.conv1x1_bn = conv_bn(class_predict_input_channels, context_enhance_channels,
                                  kernel_size=1, stride=1, pad=0)

    def forward(self, backbone_feature_tensor_odict, rpn_feature_tensor_list):

        backbone_feature_tensor_list = list(backbone_feature_tensor_odict.values())

        out_feature_tensor_odict = collections.OrderedDict()

        id_num = 0

        for backone_feature_tensor, rpn_feature_tensor in zip(backbone_feature_tensor_list, rpn_feature_tensor_list):

            AttentionClass = self.conv1x1_bn(rpn_feature_tensor)
            AttentionClass = torch.nn.functional.sigmoid(AttentionClass)
            out = backone_feature_tensor.mul(AttentionClass)

            out_feature_tensor_odict[str(id_num)] = out

            id_num = id_num + 1

        return out_feature_tensor_odict

class FeatureMergeFc(torch.nn.Module):

    def __init__(self, in_channels, representation_size, feature_size):
        super(FeatureMergeFc, self).__init__()

        self.dw_conv_full_relu = DwFullConvRelu(in_channels, representation_size, feature_size)

    def forward(self, x):

        x = self.dw_conv_full_relu(x)

        return x

class ClassAndBoxPredictor(nn.Module):

    def __init__(self, in_channels, num_classes, feature_size):
        super(ClassAndBoxPredictor, self).__init__()

        self.cls_score = FullConv(in_channels, num_classes, feature_size)
        self.bbox_pred = FullConv(in_channels, num_classes * 4, feature_size)

    def forward(self, x):

        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        scores = scores.flatten(start_dim=1)
        bbox_deltas = bbox_deltas.flatten(start_dim=1)

        return scores, bbox_deltas