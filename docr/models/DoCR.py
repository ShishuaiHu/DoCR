from torch import nn
import torch
from docr.models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from docr.models.unet import SaveFeatures, UnetBlock


class DoCR(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        cls_layers = list(resnet18(pretrained=False).children())[:-1]
        cls_base_layers = nn.Sequential(*cls_layers)
        cls_base_layers[-1] = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_branch = cls_base_layers
        self.cls_fc = nn.Linear(512, 3)

        parameter_numbers = 3 * 32 * 3 * 3 + 32
        self.controller = nn.Conv2d(3, parameter_numbers, kernel_size=1, stride=1, padding=0)
        self.bnin = nn.BatchNorm2d(32)

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        base_layers[0] = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)
        self.rec_head = nn.Conv2d(32, 3, 1)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels[l], -1, 3, 3)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels[l])

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=1,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward_cls(self, x):
        cls_features = self.cls_branch(x)
        cls_features = cls_features.view(cls_features.size(0), -1)
        cls_out = self.cls_fc(cls_features)
        return cls_out

    def gen_domain_label(self, x):
        cls_out = self.forward_cls(x)
        return F.softmax(cls_out, 1).detach()

    def forward(self, x, domain_label=None, training=False):
        domain_label = domain_label.unsqueeze(2).unsqueeze(2)
        params = self.controller(domain_label)
        params.squeeze_(-1).squeeze_(-1)

        N, _, D, H = x.size()
        head_inputs = x.reshape(1, -1, D, H)

        weight_nums, bias_nums = [], []
        weight_nums.append(3 * 32 * 3 * 3)
        bias_nums.append(32)
        dynamic_out_channels = [32]
        weights, biases = self.parse_dynamic_params(params, dynamic_out_channels, weight_nums, bias_nums)
        output = self.heads_forward(head_inputs, weights, biases, N)
        x = output.reshape(N, -1, D, H)

        x = self.bnin(x)
        x = F.relu(x)

        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))

        seg_output = self.seg_head(head_input)
        rec_output = self.rec_head(head_input)

        if training:
            return seg_output, rec_output
        else:
            return seg_output

    def close(self):
        for sf in self.sfs: sf.remove()
