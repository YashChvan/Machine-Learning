import time 
import cv2
import predict
import os
t0 = time.time()
# weight when there is no truck
base_weight = 0 #kg
weight_t0 = base_weight
weight_file = "/home/aimlgpu2/yash/weight.txt"

# expected limit above which the weight of the truck is going to be in KG
weight_expected = 45
do_prediction = True
count_till_pred = 0
temp_file = "/home/aimlgpu2/yash/text_detect/temp_detect"
temp_file_input = "/home/aimlgpu2/yash/text_detect/temp_input"
model = predict.model(temp_file)

# these two value can be changed to correct port of the cameras
front_port = 0
back_port = 1

cam_front = cv2.VideoCapture(front_port)
cam_back = cv2.VideoCapture(back_port)
# to run simulation of truck entering and exiting the station
wt = [0.0, 0.125, 0.5, 1.125, 2.0, 3.125, 4.5, 6.125, 8.0, 10.125, 12.5, 15.125, 18.0, 21.125, 24.5, 28.125, 32.0, 36.125, 40.5, 45.125, 50.0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 45.125, 40.5, 36.125, 32.0, 28.125, 24.5, 21.125, 18.0, 15.125, 12.5, 10.125, 8.0, 6.125, 4.5, 3.125, 2.0, 1.125, 0.5, 0.125, 0]

idx = 0

while True:
    if(time.time() - t0)>=0.5:
        print("checking")
        if(idx==100):
            idx = 0
        t0 = time.time()
        weight_t = 0
        
#         with open(weight_file) as f:
#             weight_t = float(f.readline().strip("\n\r"))
        weight_t = float(wt[idx])
        idx+=1
        if(abs(weight_t-base_weight)<1e-3):
            do_prediction=True
        if(abs(weight_t - weight_t0) > 1e-3):
            weight_t0 = weight_t
            continue
        #weight_t > base_weight + weight_expected and 
        #print(weight_t, t0)
        if(weight_t >= base_weight + weight_expected and do_prediction):
            if(count_till_pred < 3):
                count_till_pred+=1
                continue
            else:
                count_till_pred=0
                
            # this part can be automated but we decided not to do that
            print("enter which camera to use (front/back)")
            cam = input()
             
            # this was for testing if it works or not
            img_path = os.path.join(temp_file_input,"input.jpg")
            
            # uncomment this code so we can capture from the cams 
#             if(cam == "front"):
#                 ret, frame = cam_front.read()
#                 cv2.imwrite(img_path,frame)
#             else:
#                 ret, frame = cam_back.read()
#                 cv2.imwrite(img_path,frame)
            
            prediction = model.predict(img_path)
            print(prediction)
            do_prediction = False
        weight_t0 = weight_t
        
            
