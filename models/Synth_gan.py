import torch.nn as nn
import torch.nn.functional as F

class ImageGenerator(nn.Module):
    def __init__(self):
        super(ImageGenerator, self).__init__()

        self.up_conv7 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(2, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.up_conv7_bn = nn.BatchNorm2d(512)
        self.up_conv6 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(2, 3), stride=(2, 1), padding=(0, 1), bias=False)
        self.up_conv6_bn = nn.BatchNorm2d(512)
        self.up_conv5 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.up_conv5_bn = nn.BatchNorm2d(256)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2, 3), stride=(2, 1), padding=(0, 1), bias=False)
        self.up_conv4_bn = nn.BatchNorm2d(256)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.up_conv3_bn = nn.BatchNorm2d(256)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False)
        self.up_conv2_bn = nn.BatchNorm2d(128)
        self.up_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False)
        self.tanh1 = nn.Tanh()

        self._do_initializer()

    def _do_initializer(self):
        # he initializer(conv2d)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.normal_(tensor=module.weight, mean=0, std=0.01)

    def forward(self, inputs):
        # inputs: batch x 512 x 1 x 25
        out = self.up_conv7_bn(self.up_conv7(F.relu(inputs)))
        out = self.up_conv6_bn(self.up_conv6(F.relu(out)))
        out = self.up_conv5_bn(self.up_conv5(F.relu(out)))
        out = self.up_conv4_bn(self.up_conv4(F.relu(out)))
        out = self.up_conv3_bn(self.up_conv3(F.relu(out)))
        out = self.up_conv2_bn(self.up_conv2(F.relu(out)))
        out = self.up_conv1(F.relu(out))
        out = self.tanh1(out)

        return out


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=(2, 1), padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=(2, 1), padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=(2, 1), padding=1)
        self.sig5 = nn.Sigmoid()

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 25), stride=1)

        self._do_initializer()

    def _do_initializer(self):
        # he initializer(conv2d)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.normal_(tensor=module.weight, mean=0, std=0.01)

    def forward(self, inputs):
        # inputs: batch x 6 x 32 x 100
        out = F.leaky_relu(self.conv1(inputs), negative_slope=0.2)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2)
        out = self.sig5(self.conv5(out))
        out = self.avg_pool(out)
        out = out.view(-1)

        return out


class FeatureDiscriminator(nn.Module):
    def __init__(self):
        super(FeatureDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.sig5 = nn.Sigmoid()

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 25), stride=1)

        self._do_initializer()

    def _do_initializer(self):
        # he initializer(conv2d)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.normal_(tensor=module.weight, mean=0, std=0.01)

    def forward(self, inputs):
        # inputs: batch x 512 x 1 x 25
        out = F.leaky_relu(self.conv1(inputs))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = self.sig5(self.conv5(out))
        out = self.avg_pool(out)
        out = out.view(-1)

        return out
