import os
import cv2
import random
import json


dir = "./data/Drink_284_Detection_coco/images"
# save_dir = "./data/Drink_284_Detection_coco"
json_file = "./data/Drink_284_Detection_coco/train_coco.json"
data = json.load(open(json_file, 'r', encoding='utf-8'))
images = data['images']
categories = data['categories']

num = 8
imgs = random.sample(images, num)
for img in imgs:
    file_name = img['file_name']
    id = img['id']

    annos = [anno for anno in data['annotations'] if anno['image_id'] == id]

    img_path = os.path.join(dir, file_name)
    image = cv2.imread(img_path)

    for anno in annos:
        category_id = anno['category_id']
        label = categories[category_id]['name']
        bbox = anno['bbox']
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), thickness=5)
        cv2.putText(image, label, (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 255), thickness=5)

    # save_path = os.path.join(save_dir, file_name)
    # cv2.imwrite(save_path, image)
    cv2.namedWindow('visualization', 0)
    cv2.imshow('visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
