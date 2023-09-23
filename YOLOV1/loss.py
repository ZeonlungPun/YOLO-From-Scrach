import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self,S=7,B=2,C=20):
        super(YoloLoss,self).__init__()
        self.mse=nn.MSELoss(reduction="sum")
        self.S=S
        self.B=B
        self.C=C
        self.lambda_noobj=0.5
        self.lambda_coord=5

    def forward(self,predictions,target):
        predictions=predictions.reshape(-1,self.S,self.S,self.C+self.B*5)
        iou_b1=intersection_over_union(predictions[...,self.C+1:self.C+5],target[...,self.C+1:self.C+5])
        iou_b2=intersection_over_union(predictions[...,self.C+6:self.C+10],target[...,self.C+1:self.C+5])
        ious=torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0)
        iou_maxes,bestbox=torch.max(ious,dim=0)

        #  0 or 1 , denote whether is an object in the cell
        # (batch_size, S ,S ,1) identity_obj_i
        exists_box=target[...,self.C].unsqueeze(3)
        ##for box coordinates  bestbox=0: first boundding box is best
        # (batch,S,S,B*5+C)
        # first check which box is responsible
        box_predictions=exists_box*(
            bestbox*predictions[...,self.C+6:self.C+10]+
            (1-bestbox)*predictions[...,self.C+1:self.C+5]
        )
        box_targets=exists_box*target[...,self.C+1:self.C+5]
        # w h
        box_predictions[...,2:4]=torch.sign(box_predictions[...,2:4])*torch.sqrt(
            torch.abs(box_predictions[...,2:4]+1e-6))
        box_targets[...,2:4]=torch.sqrt(box_targets[...,2:4])
        ## (batch,S,S,B*5+C) ---> (batch*S*S,B*5+C)
        box_loss=self.mse(
            torch.flatten(box_predictions,end_dim=-2),
            torch.flatten(box_targets,end_dim=-2)
        )
        # OBJECT LOSS
        # first check which box is responsible
        pred_box=(
            bestbox*predictions[...,self.C+5:self.C+6]+(1-bestbox)*predictions[...,self.C:self.C+1]
        )
        # (batch_size, S ,S ,1) ---> (batch_size*S*S ,1)
        object_loss=self.mse(
            torch.flatten(exists_box*pred_box),
            torch.flatten(exists_box*target[...,self.C:self.C+1])
        )

        # no object loss
        # (batch_size, S ,S ,1) ---> (batch_size, S*S )
        no_object_loss1=self.mse(
            torch.flatten( (1-exists_box)*predictions[...,self.C:self.C+1],start_dim=1 ),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C + 1], start_dim=1)
        )

        no_object_loss2 = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C + 6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C + 1], start_dim=1)
        )

        #CLASS LOSS
        # (batch_size, S ,S ,C) --->(batch_size*S*S ,C)
        class_loss=self.mse(
            torch.flatten(exists_box*predictions[...,:self.C],end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )
        loss=(
            self.lambda_coord*box_loss
            +object_loss
            +self.lambda_noobj*(no_object_loss1+no_object_loss2)
            +class_loss
        )
        return loss
