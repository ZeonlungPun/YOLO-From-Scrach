import numpy as np
import os,torch
import pandas as pd
from PIL import Image,ImageFile
from torch.utils.data import DataLoader,Dataset
from utils import (iou_width_height as iou, non_max_suppression as nms)

#skip the damaged image
ImageFile.LOAD_TRUNCATED_IMAGES=True

class YOLODataset(Dataset):
    def __init__(self,
                 csv_file,
                 img_dir,
                 label_dir,
                 anchors,
                 S=[13,26,52],
                 C=20,transform=None):
        self.annotations=pd.read_csv(csv_file)
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.transform=transform
        self.S=S
        #for all 3 scales ,each scale 3 anchors
        self.anchors=torch.tensor(anchors[0]+anchors[1]+anchors[2])
        self.num_anchors=self.anchors.shape[0]
        self.num_anchors_per_scale=self.num_anchors//3
        self.C=C
        self.ignore_iou_thresh=0.5
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path=os.path.join(self.label_dir,self.annotations.iloc[index,1])
        #[class,x,y,w,h] --->  [x,y,w,h,class]
        bboxes=np.roll(np.loadtxt(fname=label_path,delimiter=" ",ndmin=2,),4,axis=1).tolist()
        img_path=os.path.join(self.img_dir,self.annotations.iloc[index,0])
        image=np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations=self.transform(image=image,bboxes=bboxes)
            image=augmentations["image"]
            bboxes=augmentations["bboxes"]
        # 6 FOR [p_o,x,y,w,h,c ]   (3,S,S,6)
        targets= [torch.zeros( (self.num_anchors//3,S,S,6) ) for S in self.S]
        """
        go through all the bounding boxes(all the objects) and assign which grid cell and anchors
        should be responsible for the object through checking highest IOU
        """
        for box in bboxes:
            iou_anchors=iou(torch.tensor(box[2:4]),self.anchors )
            #totally 9 anchors
            anchor_indices=iou_anchors.argsort(descending=True,dim=0)
            x,y,width,height,class_label=box

            #each scale should has 1 for objects
            has_anchor=[False,False,False]

            for anchor_idx in anchor_indices:
                #get which scale （grid cell）
                scale_idx=anchor_idx//self.num_anchors_per_scale # get 0,1,2
                # get the anchor from specific scale
                anchor_on_scale=anchor_idx%self.num_anchors_per_scale #get 0,1,2
                # get the grid cell num for specific feature map
                S=self.S[scale_idx]
                #find which grid cell responsing the object
                i,j=int(S*y),int(S*x) #x=0.5,s=13--> int(6.5)=6
                anchor_taken=targets[scale_idx][anchor_on_scale,i,j,0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0]=1
                    # change the coordinate to relative to the cell
                    x_cell,y_cell=S*x-j,S*y-i
                    width_cell,height_cell=width*S,height*S
                    box_coordinates=torch.tensor([x_cell,y_cell,width_cell,height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5]=box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5]=int(class_label)
                    has_anchor[scale_idx] = True

                #ignore the anchors
                elif not anchor_taken and iou_anchors[anchor_idx]>self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image,tuple(targets)





