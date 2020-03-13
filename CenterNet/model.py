import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(ResidualBlock, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.__build__()

    def __build__(self):
        self.conv1 = nn.Conv2d(self.input_feature, self.output_feature, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.output_feature, self.output_feature, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.output_feature, self.output_feature, 1, stride=1, padding=0)
        if self.input_feature != self.output_feature:
            self.conv4 = nn.Conv2d(self.input_feature, self.output_feature,3, stride=1, padding=1)

    def forward(self, x):
        init = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if self.input_feature != self.output_feature:
            init = F.relu(self.conv4(init))
        return x + init


class Hourglass(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(Hourglass, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature

        self.__build__()

    def __build__(self):
        i_f = self.input_feature
        o_f = self.output_feature

        self.down1 = ResidualBlock(i_f, o_f)
        self.down2 = ResidualBlock(o_f, o_f)
        self.down3 = ResidualBlock(o_f, o_f)
        self.down4 = ResidualBlock(o_f, o_f)
        self.down5 = ResidualBlock(o_f, o_f)

        self.skip1 = ResidualBlock(o_f, o_f)
        self.skip2 = ResidualBlock(o_f, o_f)
        self.skip3 = ResidualBlock(o_f, o_f)
        self.skip4 = ResidualBlock(o_f, o_f)

        self.middle1 = ResidualBlock(o_f, o_f)
        self.middle2 = ResidualBlock(o_f, o_f)
        self.middle3 = ResidualBlock(o_f, o_f)

        self.up1 = ResidualBlock(i_f, o_f)
        self.up2 = ResidualBlock(o_f, o_f)
        self.up3 = ResidualBlock(o_f, o_f)
        self.up4 = ResidualBlock(o_f, o_f)
        self.up5 = ResidualBlock(o_f, o_f)

    def forward(self, x):
        down1 = self.down1(x)
        skip1 = self.skip1(down1)
        down1 = F.max_pool2d(down1, (2,2))

        down2 = self.down2(down1)
        skip2 = self.skip2(down2)
        down2 = F.max_pool2d(down2, (2,2))

        down3 = self.down3(down2)
        skip3 = self.skip3(down3)
        down3 = F.max_pool2d(down3, (2,2))

        down4 = self.down4(down3)
        skip4 = self.skip4(down4)
        down4 = F.max_pool2d(down4, (2,2))

        down5 = self.down5(down4)

        middle1 = self.middle1(down5)
        middle2 = self.middle2(middle1)
        middle3 = self.middle3(middle2)

        up1 = F.interpolate(middle3, scale_factor=2)
        up1 = skip4 + up1
        up1 = self.up1(up1)

        up2 = F.interpolate(up1, scale_factor=2)
        up2 = skip3 + up2
        up2 = self.up2(up2)

        up3 = F.interpolate(up2, scale_factor=2)
        up3 = skip2 + up3
        up3 = self.up3(up3)

        up4 = F.interpolate(up3, scale_factor=2)
        up4 = skip1 + up4
        up4 = self.up4(up4)

        up5 = self.up5(up4)

        return up5


class CenterNet(nn.Module):

    def __init__(self, feature, output):
        super(CenterNet, self).__init__()
        self.feature = feature
        self.output = output

        self.__build__()

    def __build__(self):
        feature = self.feature
        self.conv7 = nn.Conv2d(3, feature, 7, stride=2, padding=3)
        self.conv3 = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.hour1 = Hourglass(feature, feature)
        self.hour2 = Hourglass(feature, feature)
        self.res1 = ResidualBlock(feature, feature)
        self.res2 = ResidualBlock(feature, feature)
        self.intermediate = nn.Conv2d(feature, self.output, 1, stride=1, padding=0)
        self.intermediate_res = ResidualBlock(self.output, feature)
        self.last1 = nn.Conv2d(feature, feature, 1, stride=1, padding=0)
        self.last2 = nn.Conv2d(feature, self.output, 1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        init = x
        x = self.hour1(x)
        res = self.res1(x)
        res = self.res2(res)
        intermediate = self.intermediate(x)
        intermediate_res = self.intermediate_res(intermediate)
        x = res + intermediate_res + init
        x = self.hour2(x)
        x = F.relu(self.last1(x))
        x = F.relu(self.last2(x))
        return intermediate, x


if __name__ == "__main__":
    net = CenterNet(256, 3)
    print(net)

    dt = torch.rand((1, 3,256,256))
    result = net(dt)
    print(result[0].size())
    print(result[1].size())

