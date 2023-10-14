import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
        self.bce=nn.BCEWithLogitsLoss()
        self.entropy=nn.CrossEntropyLoss()
        self.sigmoid=nn.Sigmoid()
        #constants
        self.lambda_class=1
        self.lambda_noobj=0.1
        self.lambda_obj=5
        self.lambda_box=10
    def forward(self,predictions,target,anchors):
        #find the index where is object and where is no object
        obj=target[...,0]==1
        noobj=target[...,0]==0

        # no object loss
        no_object_loss=self.bce(
            (predictions[...,0:1][noobj]), (target[...,0:1][noobj])  )
        # object loss
        anchors=anchors.reshape((1,3,1,1,2)) #pw *exp(tw)
        # for small change of  x y  and w ,h
        object_loss = self.bce(predictions[..., 0:1][obj],  target[..., 0:1][obj])

        # box coordinate loss
        predictions[...,1:3]=self.sigmoid(predictions[...,1:3]) # x,y must between [0,1]
        target[...,3:5]=torch.log( (1e-16+target[...,3:5])/anchors)
        box_loss=self.mse(predictions[...,1:5][obj],target[...,1:5][obj])

        # class loss
        class_loss=self.entropy(
            (predictions[...,5:][obj]), (target[...,5][obj].long()  )
        )
        return self.lambda_box*box_loss+self.lambda_obj*object_loss+self.lambda_noobj*no_object_loss+self.lambda_class*class_loss

