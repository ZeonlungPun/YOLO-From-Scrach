import albumentations as A
import numpy as np
import pandas as pd
import os,cv2
"""
crop the big image (about 2000x3000) into standard small image (640x640)
transform the bounding box labels correspondingly
additionally,do some image augmentation
"""
crop_size = 640
transform = A.Compose([
    # A.ShiftScaleRotate(),
    # A.RGBShift(),
    # A.Blur(),
    # A.GaussNoise(),
    #A.LongestMaxSize(max_size=crop_size),
    #A.PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=0),
    A.RandomCrop(width=crop_size, height=crop_size),

],bbox_params=A.BboxParams(format='yolo',label_fields=['category_ids']))

img_list=os.listdir('/home/kingargroo/seed/millet')
save_path='/home/kingargroo/seed/new_millet'
save_label_path='/home/kingargroo/seed/new_millet_label'
labels_path='/home/kingargroo/seed/millet_label'
for index,i in enumerate(img_list):
    print(' image:{}'.format(index))
    img_path = os.path.join('/home/kingargroo/seed/millet', i)
    image = cv2.imread(img_path)
    label_path = os.path.join(labels_path, i.split('.')[0])+'.txt'
    bboxes=[]
    category_ids=[]
    labels=pd.read_csv(label_path,header=None)
    for index, content in labels.iterrows():
        content_str=content[0]
        content_list = list(map(float, content_str.split(" ")))
        content_list[0] = int(content_list[0])
        cls=content_list[0]
        box=content_list[1::]
        xc,yc,w,h=box
        category_ids.append(cls)
        bboxes.append(box)
    bboxes,category_ids=np.array(bboxes),np.array(category_ids)
    for _ in range(10):
        print('image:{} for {} augmentation'.format(index,_))
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        transformed_image = transformed['image']
        print(transformed_image.shape)
        transformed_bboxes = transformed['bboxes']
        ransformed_category_ids = transformed['category_ids']
        new_img_name=i.split('.')[0]+'_'+str(_)
        new_img_save_path=os.path.join(save_path,new_img_name+'.jpg')
        cv2.imwrite(new_img_save_path,transformed_image)
        new_label_name=os.path.join(save_label_path,new_img_name+'.txt')
        txt_file = open(new_label_name, 'a+')
        for new_box in transformed_bboxes:
            xc, yc, w, h = new_box
            output_str = '0' + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) + '\n'
            txt_file.write(output_str)

