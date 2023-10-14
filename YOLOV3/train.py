import numpy as np

import config,torch,warnings
import torch.optim as optim
from model import YOLOV3
from tqdm import tqdm
from utils import mean_average_precision,cells_to_bboxes,save_checkpoint,load_checkpoint,check_class_accuracy,get_loaders,plot_couple_examples,non_max_suppression
from loss import YoloLoss
from torchmetrics.detection import MeanAveragePrecision
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def get_evaluation_bboxes(loader, model,iou_threshold,anchors,threshold, box_format="midpoint",):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x=x.to(config.DEVICE)
        # prediction is relative to a specific grid cell (normalize to the grid cell)
        predictions = model(x)
        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        true_bboxes =[[] for _ in range(batch_size)]
        # 3 scale of grid cell
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(config.DEVICE) * S
            boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
            #  want one bbox for each label, not one for each scale
            true_bbox = cells_to_bboxes( labels[i], anchor, S=S, is_preds=False)
            for idx, (box) in enumerate(true_bbox):
                true_bboxes[idx]+=box

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format)
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
            train_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes

def train_fn(train_loader,model,optimizer,loss_fn,scaled_anchors):
    loop=tqdm(train_loader,leave=True)
    losses=[]

    for batch_idx,(x,y) in enumerate(loop):
        x=x.to(config.DEVICE)
        # label for each 3 scale
        y0,y1,y2=y[0].to(config.DEVICE),y[1].to(config.DEVICE),y[2].to(config.DEVICE)

        #with torch.cuda.amp.autocast():
        out=model(x)
        loss=loss_fn(out[0],y0,scaled_anchors[0])+\
                 loss_fn(out[1],y1,scaled_anchors[1])+\
                 loss_fn(out[2],y2,scaled_anchors[2])
        losses.append(loss.item())
        optimizer.zero_grad()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        loss.backward()
        optimizer.step()

        #update progress bar
        mean_loss=sum(losses)/len(losses)
        loop.set_postfix(loss=mean_loss)




def main():
    model=YOLOV3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn=YoloLoss()
    train_loader,test_loader,train_eval_loader=get_loaders(
        train_csv_path=config.DATASET+"/8examples.csv",test_csv_path=config.DATASET+"/8examples.csv" )
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)
    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(test_loader, model, optimizer, loss_fn,  scaled_anchors)
        if config.SAVE_MODEL:
           save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
        print(f"Currently epoch {epoch}")
        print("On Train Eval loader:")
        print("On Train loader:")
        check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)
        if epoch > 0 and epoch % 1 == 0:
            print("validation start")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                    test_loader,
                    model,
                    iou_threshold=config.NMS_IOU_THRESH,
                    anchors=config.ANCHORS,
                    threshold=config.CONF_THRESHOLD,)
            mapval = mean_average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=config.MAP_IOU_THRESH,
                    box_format="midpoint",
                    num_classes=config.NUM_CLASSES, )
            print(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    main()

