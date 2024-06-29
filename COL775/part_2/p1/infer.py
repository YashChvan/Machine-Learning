import argparse
import sys
sys.path.append('./')
import torch
import regs
import models
import os
import cv2

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--normalization', type=str)
    parser.add_argument("--n",type=int)
    parser.add_argument('--test_data_file', type=str)
    parser.add_argument("--output_file",type=str)
    return parser



if __name__ == "__main__":
    parsers = get_parser()
    args = parsers.parse_args()
    model_save_dir = args.model_file
    reg = args.normalization
    
    test_dir = args.test_data_file
    output_file = args.output_file
    n = args.n
    r = 25
    
    model = models.ResNet(n,r,reg)
    model.load_state_dict(torch.load(model_save_dir))
    model.eval()
    
    imgs = {}
    valid_formats = ["jpeg","jpg","png"]
    max_id = 0
    for f in os.listdir(test_dir) :
        if(f.split(".")[-1] not in valid_formats):
            continue
        image = cv2.imread(os.path.join(test_dir,f)) 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        imgs[f.split(".")[0]] = image_rgb
        max_id = max(max_id,int(f.split(".")[0]))
    
    outputs = {}
    
    for i in imgs.keys():
        t = [imgs[i],imgs[i]]
        t = torch.tensor(t)
        t = t.permute(0,3,1,2)
        opt = model(t)
        outputs[i] = opt.argmax(dim=1)[0].item()
    
    with open(output_file,"w") as f:
        for i in range(max_id+1):
            f.write(str(outputs[i])+"\n")
        