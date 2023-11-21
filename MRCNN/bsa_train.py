import os 
import sys
import json
import datetime
import numpy as np
from tkinter import Image
from PIL import ImageFile,Image
import logging

ROOT_DIR = os.path.abspath('Mask_RCNN')
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils 

COCO_WEIGHTS_PATH=os.path.join(ROOT_DIR,'weights.h5')
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR,'logs')

class BSAconfig(Config):
    NAME = 'BSA'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 8+1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.8
    USE_MINI_MASK = False

class BSADataset(utils.Dataset):
    def load_dataset(self,dataset_dir,subset):
        self.add_class("BSA",1,"Date")
        self.add_class("BSA",2,"Narration")
        self.add_class("BSA",3,"Transaction")
        self.add_class("BSA",4,"Balance")
        self.add_class("BSA",5,"cheque")
        self.add_class("BSA",6,"serial number")
        self.add_class("BSA",7,"type")
        self.add_class("BSA",8,"description")

        assert subset in ["train","val"]
        # print("\n -------------------------------------------------------------------- \n subset:",subset)
        # print("\n______________________________________________________________________\n dataset_dir:",dataset_dir)
        dataset_dir = os.path.join(dataset_dir,subset)
        # print("________________________________________________________________________")
        # print(dataset_dir)
        annotations = json.load(open(os.path.join(dataset_dir,"75_dpi_images_project_json.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a["regions"]]
        for a in annotations:
            # print(a)
            polygons = [r['shape_attributes'] for r in a['regions']]
            class_names_str = [r['region_attributes'] for r in a['regions']]
            # print(class_names_str)
            num_ids = [str(n['column names']) for n in class_names_str]
            class_names_nums = []
            for i in num_ids:
                if i=="Date":
                    class_names_nums.append(1)
                if i=="Narration":
                    class_names_nums.append(2)
                if i=="Transaction":
                    class_names_nums.append(3)
                if i=="Balance":
                    class_names_nums.append(4)
                if i=="cheque number":
                    class_names_nums.append(5)
                if i=="serial number":
                    class_names_nums.append(6)
                if i=="type":
                    class_names_nums.append(7)
                if i=="description":
                    class_names_nums.append(8)
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            image_path = os.path.join(dataset_dir,a['filename'])
            # print(image_path)
            image = Image.open(image_path)
            width,height = image.size

            self.add_image(
                "BSA",
                image_id=a['filename'],
                path=image_path,
                width = width,height = height,
                polygons = polygons,
                class_list = np.array(class_names_nums)
            )
        
    def load_mask(self,image_id):
        image_info = self.image_info[image_id]
        if image_info["source"]!="BSA":
            return super(self.__class__,self).load_mask(image_id)
        info = self.image_info[image_id]
        mask = np.zeros([info['height'],info['width'],len(info['polygons'])],dtype='uint8')
        for i,p in enumerate(info['polygons']):
            mask[p['y']:p['y']+p['height'],p['x']:p['x']+p['width'],i]=1
        class_array = info['class_list']
        return mask,class_array
    
    def image_reference(self,image_id):
        info = self.image_info[image_id]
        if info["source"]=="BSA":
            return info["path"]
        else:
            super(self.__Class__,self).image_reference(image_id)


def train(model):
    dataset_train = BSADataset()
    dataset_train.load_dataset(r"C:\Image_annotate\75_ppi","train")
    dataset_train.prepare()

    dataset_val = BSADataset()
    dataset_val.load_dataset(r"C:\Image_annotate\75_ppi","val")
    dataset_val.prepare()
    print("Training network heads")
    model.train(dataset_train,dataset_val,
                learning_rate = Config.LEARNING_RATE,
                epochs = 50,
                layers = "heads")

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect BSA.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()
    
    if args.command == "train":
        assert args.dataset,"Argument --dataset is required for training"
    elif args.command =="splash":
        assert args.image or args.video,\
            "Provide --image or --vide to apply color splash"
        
    print("Weights: ",args.weights)
    print("Dataset: ",args.dataset)
    print("Logs: ",args.logs)

    if args.command =="train":
        config = BSAconfig()
    else:
        class InferenceConfig(BSAconfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    
    if args.command == "train":
        model = modellib.MaskRCNN(mode='training',config=config,model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference',config=config,model_dir=args.logs)
    
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower()=="last":
        weights_path = model.find_last()
    elif args.weights.lower()=="imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    print("Loading weights ",weights_path)
    if args.weights.lower()=="coco":
        model.load_weights(weights_path,by_name=True,exclude=[
            "mrcnn_class_logits","mrcnn_bbox_fc",
            "mrcnn_bbox","mrcnn_mask"
        ])
    else:
        model.load_weights(weights_path,by_name=True,exclude=[
            "mrcnn_class_logits","mrcnn_bbox_fc",
            "mrcnn_bbox","mrcnn_mask"
        ])

    if args.command =="train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "use 'train' or 'splash'".format(args.command))