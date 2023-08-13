import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from data_aug.bbox_util import *
from data_aug.data_aug import *


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
class yoloRotatebbox:
    def __init__(self, filename, image_ext, angle):
        assert os.path.isfile(filename + image_ext)
        assert os.path.isfile(filename + '.txt')

        self.filename = filename
        self.image_ext = image_ext
        self.angle = angle

        # Read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)

        # create a 2D-rotation matrix
        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
        
        
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file',type=str,default=ROOT,help="directory for output")
    parser.add_argument('--img_file',type=str,default=ROOT,help="directory of images")
    parser.add_argument('--annotation_file',type=str,default=ROOT,help="directory of annotstion")
    # parser.add_argument('--mirroring',nargs='?',const = True, default=False,help="mirror the images")
    # parser.add_argument('--random_cropping',type=int,default=0,help="random_cropping around bbox, enter the number of cropping per image")
    # parser.add_argument('--rotation',type=int,default=0,help="rotate the image, enter the maximum angle to rotate")
    # parser.add_argument('--shearing',type=int,default=0,help="shearing the image, enter maximum angle of shear")
    # parser.add_argument('--colour_shifting',type=int,default=0,help="colour shifting the images should be greater than 10")
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main():
    parse = parse_opt()
    # print(parse)
    new_file = os.path.join(parse.out_file,"augmented")
    new_img = os.path.join(new_file,"images")
    new_anot = os.path.join(new_file,"annotation")
    if(not os.path.exists(new_file)):
        os.mkdir(new_file)
    if(not os.path.exists(new_img)):
        os.mkdir(new_img)
    if(not os.path.exists(new_anot)):
        os.mkdir(new_anot)
    # print(new_file)
    img_dir = parse.img_file
    anot_dir = parse.annotation_file
    
    rot_angles = [15,30,45]
    shear_factor = [0.1,0.3,0.5]
    randm_translation = [0.3,0.5]
    HSV_trans = [(100,100,100)]
    file_formats = ['jpeg','png','jpg']
    for file in os.listdir(img_dir):
        print(file)
        if(file.split(".")[-1] in file_formats):
            img = cv2.imread(os.path.join(img_dir,file))
            dh, dw, _ = img.shape
            # print(img)
            bboxs = []
            with open(os.path.join(anot_dir,file.split(".")[0]+".txt")) as f:
                for t in f.readlines():
                    t = t.rstrip("\n\r")
                    c, x, y, w, h = map(float, t.split(' '))
                    # l = l.split(" ")
                    l = int((x - w / 2) * dw)
                    r = int((x + w / 2) * dw)
                    t = int((y - h / 2) * dh)
                    b = int((y + h / 2) * dh) 
                    bboxs.append([l,t,r,b,c])
            
            bboxs = np.array(bboxs)
            
            # adding the same image
            cv2.imwrite(os.path.join(new_img,file.split(".")[0]+".jpg"),img)
            bbox_ = to_yolo(img.copy(),bboxs.copy())
            bbox_ = bbox_.astype(str).tolist()
            with open(os.path.join(new_anot,file.split(".")[0]+".txt"),'w+') as f:
                for i in range(len(bboxs)):
                    # print(changed[i])
                    f.write(" ".join(bbox_[i])+"\n")
                    
            # horizontal filp transformation     
            img_, bbox_ = RandomHorizontalFlip(1)(img,bboxs)
            cv2.imwrite(os.path.join(new_img,file.split(".")[0]+"_mirrored"+".jpg"),img_)
            bbox_ = to_yolo(img_.copy(),bbox_.copy())
            bbox_ = bbox_.astype(str).tolist()
            with open(os.path.join(new_anot,file.split(".")[0]+"_mirrored"+".txt"),'w+') as f:
                for i in range(len(bbox_)):
                    # print(changed[i])
                    f.write(" ".join(bbox_[i])+"\n")
            
            #for rotation
            count = 1
            for ang in rot_angles:
                img_, bbox_ = RandomRotate((ang,ang))(img,bboxs)
                cv2.imwrite(os.path.join(new_img,file.split(".")[0]+"_rotated_"+str(count)+".jpg"),img_)
                bbox_ = to_yolo(img_.copy(),bbox_.copy())
                bbox_ = bbox_.astype(str).tolist()
                with open(os.path.join(new_anot,file.split(".")[0]+"_rotated_"+str(count)+".txt"),'w+') as f:
                    for i in range(len(bbox_)):
                        # print(changed[i])
                        f.write(" ".join(bbox_[i])+"\n")
                count += 1
                img_, bbox_ = RandomRotate((-ang,-ang))(img,bboxs)
                cv2.imwrite(os.path.join(new_img,file.split(".")[0]+"_rotated_"+str(count)+".jpg"),img_)
                bbox_ = to_yolo(img_.copy(),bbox_.copy())
                bbox_ = bbox_.astype(str).tolist()
                with open(os.path.join(new_anot,file.split(".")[0]+"_rotated_"+str(count)+".txt"),'w+') as f:
                    for i in range(len(bbox_)):
                        # print(changed[i])
                        f.write(" ".join(bbox_[i])+"\n")
                count += 1
                
            # for shearing 
            count = 1
            for fac in shear_factor:
                img_, bbox_ = RandomShear((fac,fac))(img,bboxs)
                cv2.imwrite(os.path.join(new_img,file.split(".")[0]+"_sheared_"+str(count)+".jpg"),img_)
                bbox_ = to_yolo(img_.copy(),bbox_.copy())
                bbox_ = bbox_.astype(str).tolist()
                with open(os.path.join(new_anot,file.split(".")[0]+"_sheared_"+str(count)+".txt"),'w+') as f:
                    for i in range(len(bbox_)): 
                        # print(changed[i])
                        f.write(" ".join(bbox_[i])+"\n")
                count += 1
                img_, bbox_ = RandomShear((-fac,-fac))(img,bboxs)
                cv2.imwrite(os.path.join(new_img,file.split(".")[0]+"_sheared_"+str(count)+".jpg"),img_)
                bbox_ = to_yolo(img_.copy(),bbox_.copy())
                bbox_ = bbox_.astype(str).tolist()
                with open(os.path.join(new_anot,file.split(".")[0]+"_sheared_"+str(count)+".txt"),'w+') as f:
                    for i in range(len(bbox_)):
                        # print(changed[i])
                        f.write(" ".join(bbox_[i])+"\n")
                count += 1
            
            # for random translations
            count = 1
            for fac in randm_translation:
                img_, bbox_ = RandomTranslate((fac,fac))(img,bboxs)
                cv2.imwrite(os.path.join(new_img,file.split(".")[0]+"_translated_"+str(count)+".jpg"),img_)
                bbox_ = to_yolo(img_.copy(),bbox_.copy())
                bbox_ = bbox_.astype(str).tolist()
                with open(os.path.join(new_anot,file.split(".")[0]+"_translated_"+str(count)+".txt"),'w+') as f:
                    for i in range(len(bbox_)): 
                        # print(changed[i])
                        f.write(" ".join(bbox_[i])+"\n")
                count += 1
                # img_, bbox_ = RandomTranslate((-fac,-fac))(img,bboxs)
                # cv2.imwrite(os.path.join(new_img,file.split(".")[0]+"_translated_"+str(count)+".jpg"),img_)
                # bbox_ = to_yolo(img_.copy(),bbox_.copy())
                # bbox_ = bbox_.astype(str).tolist()
                # with open(os.path.join(new_anot,file.split(".")[0]+"_translated_"+str(count)+".txt"),'w+') as f:
                #     for i in range(len(bbox_)):
                #         # print(changed[i])
                #         f.write(" ".join(bbox_[i])+"\n")
                # count += 1
            # for hew changes
            count = 1
            for fac in HSV_trans:
                img_, bbox_ = RandomHSV(fac[0],fac[1],fac[2])(img,bboxs)
                cv2.imwrite(os.path.join(new_img,file.split(".")[0]+"_HSV_"+str(count)+".jpg"),img_)
                bbox_ = to_yolo(img_.copy(),bbox_.copy())
                bbox_ = bbox_.astype(str).tolist()
                with open(os.path.join(new_anot,file.split(".")[0]+"_HSV_"+str(count)+".txt"),'w+') as f:
                    for i in range(len(bbox_)): 
                        # print(changed[i])
                        f.write(" ".join(bbox_[i])+"\n")
                count += 1

            
    
                
if __name__ == "__main__":
    main()