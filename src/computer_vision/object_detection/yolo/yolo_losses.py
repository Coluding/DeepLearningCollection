import torch
import torch.nn as nn


class Yolov1Loss(nn.Module):
    def __init__(self, l_coord: float, l_noobj: float):
        super().__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, gt):
        # ================#
        #  Object Loss   #
        # ================#
        objectness_one = gt[:, :, :, 4].eq(1).float()
        x_loss = self.mse(gt[:, :, :, 0], pred[:, :, :, 0]) * objectness_one
        y_loss = self.mse(gt[:, :, :, 1], pred[:, :, :, 1]) * objectness_one
        w_loss = self.mse(torch.sqrt(gt[:, :, :, 2] + 1e-8), torch.sqrt(pred[:, :, :, 2] + 1e-8)) * objectness_one
        h_loss = self.mse(torch.sqrt(gt[:, :, :, 3] + 1e-8), torch.sqrt(pred[:, :, :, 3] + 1e-8)) * objectness_one
        obj_loss = self.mse(gt[:, :, :, 4], pred[:, :, :, 4]) * objectness_one
        coord_loss = self.l_coord * torch.sum(x_loss + y_loss) + self.l_coord * torch.sum(w_loss + h_loss + obj_loss)
        # print(f"Coord loss: {coord_loss:.4f}")

        # ================#
        #  No-Object Loss   #
        # ================#

        noobject_one = gt[:, :, :, 4].eq(0).float()
        noobj_loss = self.mse(gt[:, :, :, 4], pred[:, :, :, 4]) * noobject_one
        noobj_loss = self.l_noobj * torch.sum(noobj_loss)
        # print(f"Noobj loss: {noobj_loss:.2f}")

        # ================#
        #  Class Loss   #
        # ================#
        class_loss = torch.sum(self.mse(gt[:, :, :, 5:], pred[:, :, :, 5:]))
        # print(f"Class loss: {class_loss}")

        return coord_loss + noobj_loss + class_loss