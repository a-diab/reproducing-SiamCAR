import torch
import torch.nn as nn
import torch.nn.functional as functional

from Resnet50_Backbone import resnet50
from heads import Heads
from loss_car import make_siamcar_loss_evaluator

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # backbone
        self.backbone = resnet50([2, 3, 4])
        # head
        self.head = Heads(256)
        # dimension reduction
        self.reduce = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

        # loss functions - taken from SiamCAR
        self.loss_evaluator = make_siamcar_loss_evaluator()

        # logsoftmax
        self.logsoftmax = nn.LogSoftmax()

    def dwXcorr(self, x, kernel):
        # depthwise cross correlation, same as SiamRPN++
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = functional.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    # Pysot utils loacation_grid from SiamCAR
    def compLevelLocs(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + 32
        return locations

    def forward(self, data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        Z = self.backbone(template)
        X = self.backbone(search)

        fmap = self.dwXcorr(X[0], Z[0])
        for i in range(len(X)-1):
            newfmap = self.dwXcorr(X[i+1],self.Z[i+1])
            fmap = torch.cat([fmap,newfmap],1)
        fmap = self.reduce(fmap)

        cls, loc, cen = self.head(fmap)

        h, w = fmap.size()[-2:]
        locs = self.compLevelLocs(
            h, w, 8,
            fmap.device
        )

        cls = self.logsoftmax(cls)

        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locs,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = 1 * cls_loss + \
            2 * loc_loss + 1 * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs




