import timm
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torchvision import models


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##############################
# 原有模块（保持不变）
##############################

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class Residual(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(Residual, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class BCG(nn.Module):
    def __init__(self, in_dim):
        super(BCG, self).__init__()
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, boundary):
        m_batchsize, C, height, width = x.size()
        q = self.query_conv1(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k = self.key_conv1(boundary).view(m_batchsize, -1, width * height)
        v = self.value_conv1(x).view(m_batchsize, -1, width * height)
        energy1 = torch.bmm(q, k)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
        out1 = x + out1
        return out1


class CotSR(nn.Module):
    def __init__(self, in_dim):
        super(CotSR, self).__init__()
        self.chanel_in = in_dim
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        m_batchsize, C, height, width = x1.size()
        q1 = self.query_conv1(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width * height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width * height)
        q2 = self.query_conv2(x2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width * height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width * height)
        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2
        return out1, out2


class MSD(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(MSD, self).__init__()
        self.out_channels = out_channels
        self.input_channels = input_channels
        self.layer1 = BasicConv2d(input_channels, out_channels, 1)
        self.layer2 = BasicConv2d(out_channels, out_channels, 3, 1, 1, 1)
        self.layer3 = BasicConv2d(out_channels, out_channels, 5, 1, 2, 1)

    def forward(self, edge):
        x = self.layer1(edge)
        x1 = edge + x
        x2 = self.layer2(x1)
        x3 = x2 + x1
        x4 = self.layer3(x3)
        x_total = x4 + x3
        return x_total


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


##############################
# 新增模块1: 语义图节点构建
##############################

class SemanticGraphNode(nn.Module):
    def __init__(self, in_channels, num_nodes=8):
        super(SemanticGraphNode, self).__init__()
        self.num_nodes = num_nodes
        self.node_assign = nn.Sequential(
            nn.Conv2d(in_channels, num_nodes, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_nodes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assign = self.node_assign(x)
        assign_flat = assign.view(B, self.num_nodes, -1)
        x_flat = x.view(B, C, -1)
        nodes = torch.bmm(assign_flat, x_flat.permute(0, 2, 1))
        node_count = assign_flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        nodes = nodes / node_count
        return nodes, assign


##############################
# 新增模块2: 图注意力交互
##############################

class GraphInteraction(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(GraphInteraction, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )
        self.norm2 = nn.LayerNorm(channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, nodes):
        B, N, C = nodes.shape
        residual = nodes
        nodes = self.norm(nodes)

        q = self.q_proj(nodes).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(nodes).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(nodes).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        nodes = residual + out

        residual = nodes
        nodes = self.norm2(nodes)
        nodes = residual + self.ffn(nodes)

        return nodes


##############################
# 新增模块3: 层次化语义图交互（HGM）
##############################

class HierarchicalGraphModule(nn.Module):
    def __init__(self, low_channels, high_channels, num_nodes=8):
        super(HierarchicalGraphModule, self).__init__()
        self.num_nodes = num_nodes
        self.low_channels = low_channels
        self.high_channels = high_channels
        self.unified_channels = low_channels

        if high_channels != low_channels:
            self.high_proj = nn.Sequential(
                nn.Conv2d(high_channels, low_channels, 1, bias=False),
                nn.BatchNorm2d(low_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.high_proj = nn.Identity()

        self.low_node_builder = SemanticGraphNode(low_channels, num_nodes)
        self.high_node_builder = SemanticGraphNode(low_channels, num_nodes)

        self.low_graph_interact = GraphInteraction(low_channels, num_heads=4)
        self.high_graph_interact = GraphInteraction(low_channels, num_heads=4)

        self.cross_q = nn.Linear(low_channels, low_channels)
        self.cross_k = nn.Linear(low_channels, low_channels)
        self.cross_v = nn.Linear(low_channels, low_channels)
        self.cross_norm_q = nn.LayerNorm(low_channels)
        self.cross_norm_kv = nn.LayerNorm(low_channels)
        self.cross_scale = (low_channels // 4) ** -0.5

        self.low_back_proj = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, 1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
        )
        if high_channels != low_channels:
            self.high_back_proj = nn.Sequential(
                nn.Conv2d(low_channels, high_channels, 1, bias=False),
                nn.BatchNorm2d(high_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.high_back_proj = nn.Sequential(
                nn.Conv2d(low_channels, low_channels, 1, bias=False),
                nn.BatchNorm2d(low_channels),
                nn.ReLU(inplace=True)
            )

        self.low_scale = nn.Parameter(torch.zeros(1))
        self.high_scale = nn.Parameter(torch.zeros(1))

    def node_to_spatial(self, nodes, assign, H, W):
        B, N, C = nodes.shape
        assign_flat = assign.view(B, N, -1)
        spatial = torch.bmm(nodes.permute(0, 2, 1), assign_flat)
        spatial = spatial.view(B, C, H, W)
        return spatial

    def forward(self, x_low, x_high):
        B = x_low.shape[0]
        H_low, W_low = x_low.shape[2], x_low.shape[3]
        H_high, W_high = x_high.shape[2], x_high.shape[3]

        x_high_unified = self.high_proj(x_high)

        low_nodes, low_assign = self.low_node_builder(x_low)
        high_nodes, high_assign = self.high_node_builder(x_high_unified)

        low_nodes = self.low_graph_interact(low_nodes)
        high_nodes = self.high_graph_interact(high_nodes)

        cross_residual = high_nodes
        q = self.cross_q(self.cross_norm_q(high_nodes))
        k = self.cross_k(self.cross_norm_kv(low_nodes))
        v = self.cross_v(self.cross_norm_kv(low_nodes))
        cross_attn = (q @ k.transpose(-2, -1)) * self.cross_scale
        cross_attn = cross_attn.softmax(dim=-1)
        cross_out = cross_attn @ v
        high_nodes = cross_residual + cross_out

        low_spatial = self.node_to_spatial(low_nodes, low_assign, H_low, W_low)
        low_spatial = self.low_back_proj(low_spatial)

        high_spatial = self.node_to_spatial(high_nodes, high_assign, H_high, W_high)
        high_spatial = self.high_back_proj(high_spatial)

        x_low_enhanced = x_low + self.low_scale * low_spatial
        x_high_enhanced = x_high + self.high_scale * high_spatial

        return x_low_enhanced, x_high_enhanced


##############################
# 新增模块4: 差异特征增强（DE）
##############################

class DifferenceEnhancement(nn.Module):
    def __init__(self, in_channels):
        super(DifferenceEnhancement, self).__init__()
        self.diff_conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.diff_conv3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.diff_conv5 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        self.enhance_scale = nn.Parameter(torch.zeros(1))

    def forward(self, t1_feat, t2_feat):
        diff_abs = torch.abs(t1_feat - t2_feat)
        concat_feat = torch.cat([t1_feat, t2_feat], dim=1)
        d1 = self.diff_conv1(concat_feat)
        d3 = self.diff_conv3(concat_feat)
        d5 = self.diff_conv5(concat_feat)
        d_fuse = self.fuse(torch.cat([d1, d3, d5], dim=1))
        ca = self.channel_attn(d_fuse)
        d_fuse = d_fuse * ca
        out = diff_abs + self.enhance_scale * d_fuse
        return out


##############################
# 新增模块5: 语义精炼（SR）
##############################

class SemanticRefine(nn.Module):
    def __init__(self, channels):
        super(SemanticRefine, self).__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.refine_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.refine_scale * self.refine(x)


##############################
# SRDF 主网络（带消融控制开关）
##############################

class SRDF(nn.Module):
    def __init__(self, channel=32, num_classes=7, pretrained_path=None, drop_rate=0.4,
                 use_hgm=True, use_de=True, use_sr=True):
        """
        Args:
            channel: 基础通道数
            num_classes: 语义类别数
            pretrained_path: PVT预训练权重路径
            drop_rate: dropout率
            use_hgm: 是否启用 HierarchicalGraphModule（层次化语义图交互）
            use_de:  是否启用 DifferenceEnhancement（差异特征增强）
            use_sr:  是否启用 SemanticRefine（语义特征精炼）
        """
        super(SRDF, self).__init__()

        # 保存消融开关
        self.use_hgm = use_hgm
        self.use_de = use_de
        self.use_sr = use_sr

        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()

        # 加载预训练权重
        if pretrained_path is not None:
            save_model = torch.load(pretrained_path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        self.channel = channel
        self.num_classes = num_classes

        # 通道转换层
        self.Translayer1_1 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel * 2, 1)
        self.Translayer4_1 = BasicConv2d(512, channel * 2, 1)
        self.concat_channels = channel + channel * 2
        self.Translayer5_1 = BasicConv2d(self.concat_channels, channel, 1)

        # 上采样层
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 输出头
        self.out_feature = nn.Conv2d(channel, num_classes, 1)
        self.out_feature2 = nn.Conv2d(channel, num_classes, 1)
        self.out_feature3 = nn.Conv2d(channel, 1, kernel_size=1)
        self.out_feature1 = nn.Conv2d(channel, 1, 1)

        # CBAM
        self.C0 = CBAM(channel)
        self.C1 = CBAM(channel * 2)

        # 原有功能模块
        self.ms = MSD(channel, channel)
        self.cot = CotSR(channel)
        self.bcg = BCG(channel)
        self.cov16 = BasicConv2d(channel, channel, 3, 1, 1, 1)

        # ====== 条件创建新增模块 ======
        if self.use_hgm:
            self.hierarchical_graph = HierarchicalGraphModule(
                low_channels=channel,
                high_channels=channel * 2,
                num_nodes=8
            )

        if self.use_de:
            self.diff_enhance = DifferenceEnhancement(channel)

        if self.use_sr:
            self.sem_refine1 = SemanticRefine(channel)
            self.sem_refine2 = SemanticRefine(channel)

        # 打印启用状态
        print(f"=== BGSNet Ablation Config ===")
        print(f"  HierarchicalGraphModule (HGM): {'ON' if use_hgm else 'OFF'}")
        print(f"  DifferenceEnhancement   (DE):  {'ON' if use_de else 'OFF'}")
        print(f"  SemanticRefine          (SR):  {'ON' if use_sr else 'OFF'}")
        print(f"==============================")

    def base_forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]
        x1 = self.drop(x1)
        x2 = pvt[1]
        x2 = self.drop(x2)
        x3 = pvt[2]
        x3 = self.drop(x3)
        x4 = pvt[3]
        x4 = self.drop(x4)

        x1 = self.Translayer1_1(x1)
        x2_2 = self.Translayer2_1(x2)
        x2_3 = self.up1(x2_2)

        x_low = x1 + x2_3
        x_low = self.C0(x_low)

        x3_2 = self.Translayer3_1(x3)
        x4_2 = self.Translayer4_1(x4)
        x4_3 = self.up2(x4_2)

        x_high = x3_2 + x4_3
        x_high = self.C1(x_high)

        # 【消融开关】HGM
        if self.use_hgm:
            x_low, x_high = self.hierarchical_graph(x_low, x_high)

        x_total = torch.cat((x_low, self.up3(x_high)), 1)
        x_fuse = self.Translayer5_1(x_total)

        return x_fuse

    def forward(self, t1, t2):
        t1_out = self.base_forward(t1)
        t2_out = self.base_forward(t2)

        t1_out, t2_out = self.cot(t1_out, t2_out)

        # 【消融开关】DE
        if self.use_de:
            changes = self.diff_enhance(t1_out, t2_out)
        else:
            changes = torch.abs(t2_out - t1_out)

        # 边界提取（始终使用原始abs diff）
        b_changes = torch.abs(t2_out - t1_out)
        boundary = self.ms(b_changes)
        out_edge = self.out_feature3(boundary)

        change_areas = self.cov16(changes)
        out_c = self.bcg(change_areas, boundary)
        out_cs = self.out_feature1(out_c)

        # 【消融开关】SR
        if self.use_sr:
            t1_refined = self.sem_refine1(t1_out)
            t2_refined = self.sem_refine2(t2_out)
        else:
            t1_refined = t1_out
            t2_refined = t2_out

        output1 = self.out_feature(t1_refined)
        output2 = self.out_feature2(t2_refined)

        prediction1_1 = F.interpolate(out_cs, scale_factor=4, mode='bilinear')
        prediction1_2 = F.interpolate(output1, scale_factor=4, mode='bilinear')
        prediction1_3 = F.interpolate(output2, scale_factor=4, mode='bilinear')
        predict_edge = F.interpolate(out_edge, scale_factor=4, mode='bilinear')

        return prediction1_1, prediction1_2, prediction1_3, predict_edge


