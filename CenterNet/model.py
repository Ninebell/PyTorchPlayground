import torch
import torch.nn as nn
import torch.nn.functional as F
from CenterNet.data_util import data_generator, check_iou, calc_inter
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw, Image

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

        self.batch1 = nn.BatchNorm2d(self.output_feature)
        self.batch2 = nn.BatchNorm2d(self.output_feature)
        self.batch3 = nn.BatchNorm2d(self.output_feature)

    def forward(self, x):
        init = x
        x = torch.relu(self.batch1(self.conv1(x)))
        x = torch.relu(self.batch2(self.conv2(x)))
        x = torch.relu(self.batch3(self.conv3(x)))
        if self.input_feature != self.output_feature:
            init = torch.relu(self.conv4(init))
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
        self.conv7 = nn.Conv2d(3, feature, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.hour1 = Hourglass(feature, feature)
        # self.hour2 = Hourglass(feature, feature)
        self.res1 = ResidualBlock(feature, feature)
        self.res2 = ResidualBlock(feature, feature)
        # self.intermediate = nn.Conv2d(feature, self.output, 1, stride=1, padding=0)
        # self.intermediate_res = ResidualBlock(self.output, feature)
        self.heat_last_f = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.heat_last = nn.Conv2d(feature, 1, 1, stride=1, padding=0)

        self.size_last_f = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.size_last = nn.Conv2d(feature, 2, 1, stride=1, padding=0)
        self.batch1 = nn.BatchNorm2d(feature)
        self.batch2 = nn.BatchNorm2d(feature)
        self.batch3 = nn.BatchNorm2d(feature)
        self.batch4 = nn.BatchNorm2d(feature)

    def forward(self, x):
        x = torch.relu(self.batch1(self.conv7(x)))
        x = F.max_pool2d(torch.relu(self.batch2(self.conv3(x))), (2,2))
        # init = x
        x = self.hour1(x)
        res = self.res1(x)
        res = self.res2(res)

        heat = torch.relu(self.batch3(self.heat_last_f(res)))
        heat = torch.sigmoid(self.heat_last(heat))

        size = torch.relu(self.batch4(self.size_last_f(res)))
        size = torch.relu(self.size_last(size))
        # intermediate = self.intermediate(x)
        # intermediate_res = self.intermediate_res(intermediate)
        # x = res + intermediate_res + init
        # x = self.hour2(x)
        # x = F.relu(self.last1(x))
        # x = F.relu(self.last2(x))
        return heat, size


def focal_loss(output, target):

    ones = torch.ones((64,64)).cuda()
    zeros = torch.zeros((64,64)).cuda()

    ones_board = torch.where(target == 1, output, ones)
    zeros_board = torch.where(target != 1, output, zeros)

    alpha = 2
    beta = 4

    N = torch.where(target == 1, target, zeros)
    N = torch.sum(N)

    epsilon = 1e-10

    ones_board = torch.pow(1-ones_board, alpha) * torch.log(ones_board+epsilon)
    zeros_board = torch.pow(1-target, beta) * torch.pow(zeros_board, alpha) * torch.log(1-zeros_board+epsilon)

    return -(ones_board+zeros_board).sum()/N


def size_loss(output, target, center):

    epsilon = 1e-10
    zeros = torch.zeros((64,64)).cuda()
    ones = torch.ones((64,64)).cuda()

    N = torch.where(center == 1, center, zeros)
    N = torch.sum(N)

    l1_output = torch.where(center == 1, output, zeros)
    log_output = torch.where(center == 1, output, ones)

    l1_loss = torch.abs(l1_output-target)
    # log_loss = torch.abs(torch.log(log_output+epsilon)-torch.log(target+epsilon))

    l1 = l1_loss.sum()/N * 0.1
    logl1 = torch.log(1-torch.tanh(l1_loss)+epsilon).sum()/N

    return l1 - logl1
    # return l1
    # return logl1


def center_loss(output, target):
    o_heat = output[0]
    o_size = output[1]
    t_heat = torch.from_numpy(target[0]).type(torch.FloatTensor).cuda()
    t_size = torch.from_numpy(target[1]).type(torch.FloatTensor).cuda()

    fl = focal_loss(o_heat, t_heat)

    sz = size_loss(o_size, t_size, t_heat)
    return fl + sz


def draw_roi(img, heat, size):
    heat = heat[0]
    img = np.asarray(img*255,dtype=np.uint8)
    img = Image.fromarray(img)
    img_draw = ImageDraw.Draw(img)

    center = []
    for r in range(1,63):
        for c in range(1,63):
            if heat[r,c] == np.max(heat[r-1:r+2, c-1:c+2]) and heat[r,c] > 0.5:
                center.append((c,r))

    for point in center:
        w = size[0,point[1],point[0]] / 592 * 256
        h = size[1,point[1],point[0]] / 480 * 256
        point = point[0]*4, point[1]*4
        img_draw.rectangle((point[0]-w//2, point[1]-h//2, point[0]+w//2, point[1]+h//2), outline='red', width=1)

    return img


def make_roi(heat, size):
    rois = []
    for r in range(1,heat.shape[0]-1):
        for c in range(1,heat.shape[1]-1):
            if heat[r,c] == np.max(heat[r-1:r+2, c-1:c+2]) and heat[r,c] > 0.5:
                w = size[0, r,c] / 592 * 256
                h = size[1, r,c] / 480 * 256
                rois.append((c - w/2,r - h/2, w, h))

    return rois


if __name__ == "__main__":
    epoches = 1000
    min_loss = 100009.47

    net = CenterNet(256, 3)

    criterion = center_loss

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        net = net.cuda()

    lr = 1e-4

    losses = []
    ioues = []

    optim = torch.optim.Adam(net.parameters(), lr)

    # net.load_state_dict(torch.load('./models/min_loss{0:.2f}.dict'))

    for epoch in range(epoches):
        iou_count = 0
        epoch_iou = 0
        epoch_loss = 0
        for data in tqdm(data_generator(8, shuffle=True, is_train=True)):
            x, heat, size = data
            if is_cuda:
                x = torch.from_numpy(x).type(torch.FloatTensor).cuda()

            optim.zero_grad()
            result = net(x)
            loss = criterion(result, [heat, size])
            loss.backward()
            epoch_loss += loss.item()


            with torch.no_grad():
                label_rois = make_roi(heat[0][0], size[0])
                predict_rois = make_roi(result[0][0][0].cpu().numpy(), result[1][0].cpu().numpy())
                for label in label_rois:
                    for predict in predict_rois:
                        iou = check_iou(label, predict)
                        if iou >= 0.5:
                            iou_count = iou_count + 1
                            epoch_iou +=iou
            optim.step()


        if iou_count == 0:
            print(epoch, epoch_loss, 0)
        else:
            print(epoch, epoch_loss, epoch_iou/iou_count)

        losses.append(loss.item())
        if iou_count == 0:
            ioues.append(0)
        else:
            ioues.append(epoch_iou/iou_count)
        sleep(0.1)

        if min_loss > epoch_loss:
            min_loss = epoch_loss
            torch.save(net.state_dict(), './models/min_loss{0:.2f}.dict'.format(min_loss))

    net.load_state_dict(torch.load('./models/min_loss{0:.2f}.dict'.format(min_loss)))
    plt.subplot(2,1,1)
    plt.plot(np.arange(epoches),losses)
    plt.subplot(2,1,2)
    plt.plot(np.arange(epoches),ioues)
    plt.savefig('loss.png')
    plt.show()


    count = 0
    for data in data_generator(1, False, False):
        x, heat, size = data
        if is_cuda:
            x = torch.from_numpy(x).type(torch.FloatTensor).cuda()

        with torch.no_grad():
            result = net(x)

            input_img = x[0].cpu().numpy()
            heat_img = heat[0]

            input_img = np.moveaxis(input_img, 0, 2)
            heat_img = np.reshape(heat_img,(64,64))
            size_img = draw_roi(input_img, heat[0], size[0])

            predict_heat = np.reshape(result[0][0].cpu().numpy(), (64,64))
            predict_size = draw_roi(input_img, result[0][0].cpu().numpy(), result[1][0].cpu().numpy())

            plt.subplot(3,2,1)
            plt.imshow(input_img)

            plt.subplot(3,2,3)
            plt.imshow(heat_img)

            plt.subplot(3,2,4)
            plt.imshow(predict_heat)

            plt.subplot(3,2,5)
            plt.imshow(size_img)

            plt.subplot(3,2,6)
            plt.imshow(predict_size)
            plt.savefig('{0}.png'.format(count))

            label_rois = make_roi(heat[0][0], size[0])
            predict_rois = make_roi(result[0][0][0].cpu().numpy(), result[1][0].cpu().numpy())

            iou_sum = 0
            iou_count = 0
            for label in label_rois:
                for predict in predict_rois:
                    iou = check_iou(label, predict)
                    if iou >= 0.5:
                        iou_count = iou_count + 1
                        iou_sum +=iou

            count = count + 1
            print(iou_count, iou_sum/iou_count)

            if count > 10:
                break
            plt.show()
