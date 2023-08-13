import mmocr
from mmocr.apis import MMOCRInferencer
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
class model():
    def __init__(self, output_path):
        self.out_p1_path = output_path
        self.img_ori = None
        self.infer1 = MMOCRInferencer(rec='NRTR',device="cpu")
        self.infer2 = MMOCRInferencer(rec='SAR',device="cpu")
        self.infer3 = MMOCRInferencer(rec='SVTR-base',device="cpu")
        
    # to preforn image segmentation
    def clean_img(self,img_path):
        self.img_ori  = cv2.imread(str(img_path))
        # python3 test.py --checkpoint /Users/yashchavan/Programming/python/MASTER-pytorch-main/configs/config.json --img_folder /Users/yashchavan/Desktop/Machine_learning/License_plate_project/text_detect/croped_text --width 160 --height 48 --output_folder /Users/yashchavan/Desktop/Machine_learning/License_plate_project/text_detect/output --gpu 0 --batch_size 64
        # print(self.img_ori)
        final_img = cv2.imread(str(img_path))
        final_img = cv2.resize(final_img,(256,64),interpolation=cv2.INTER_LINEAR)
        self.img_ori  = cv2.resize(self.img_ori,(512,64),interpolation=cv2.INTER_LINEAR)
        img = self.img_ori.copy()
        img_new  = np.zeros((img.shape[0],img.shape[1]))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        avg = np.average(img_gray)
        # print(avg)
        alpha = 1.5*(68)/max(avg,68) # Contrast control
        beta = 10*68/max(avg,68) # Brightness control
        img= cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]): 
                if(img[i][j][0]<=120 and img[i][j][1]<=120 and img[i][j][2]<=120):
                    img_new[i][j] = 255

        sum_hor = img_new.sum(axis=1)
        sum_hor = 10*sum_hor/np.max(sum_hor)
        x_axis = np.arange(sum_hor.shape[0])
        m1 = sum_hor.shape[0]//2
        m2 = m1
        has_two_lines = False
        if(np.min(sum_hor[m1-5:m1+5]) <= 3):
            has_two_lines = True
        sub_imgs = []
        if has_two_lines:
            img1 = final_img[0:int(0.6*sum_hor.shape[0]),:]
            img2 = final_img[int(0.4*sum_hor.shape[0]):-1 , :]
            
            sub_imgs.append(img1)
            sub_imgs.append(img2)
        else:
            sub_imgs.append(final_img)

        # print(img1.shape,img2.shape)
        print("num of splits : ",len(sub_imgs))
        for i in range(len(sub_imgs)):
            cv2.imwrite(os.path.join(self.out_p1_path,str(i)+".jpg"), sub_imgs[i])
            # cv2.imshow("img",sub_imgs[i])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    # to remove any unwanted characters from the prediction
    def get_valid(self,s):
        val = ['A',"B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","1","2","3","4","5","6","7","8","9","0"]
        temp = ""
        for i in s:
            if str(i) in val:
                temp+=str(i)
        return temp
    # to get what type of license plate the image is
    def get_type(self,s):
        char_seg_count = 0
        prev_i = 0
        number = ["0","1","2","3","4","5","6","7","8","9"]
        for i in range(1,len(s)):
            if(s[i] not in number and s[i-1] not in number):
                continue
            elif (s[i] in number and s[i-1] not in number):
                char_seg_count += 1
            else:
                continue

        if(char_seg_count ==1):
            return 2
        else:
            return 1
    # to rectify the errors in type 1 license plate(new format)
    def repair_type_1(self,s):
        state_code = ["AN","AP","AR","AS","BH","BR","CH","CG","DD","DL","GA","GJ","HR","HP","JK","JH","KA","KL","LA","LD","MP","MH","MN","ML","MZ","NL","OD","PY","PB","RJ","SK","TN","TS","TR","UP","UK","WB"]
        number = ["0","1","2","3","4","5","6","7","8","9"]
        repaired_s = []
        for i in s:
            repaired_s.append(i)

        for i in range(2,len(s)):
            if(repaired_s[i] == 'O'):
                repaired_s[i] = "0"
            if(repaired_s[i] == "I"):
                repaired_s[i] = "1"
        np_st_code = repaired_s[0]+repaired_s[1]
        np_ditrict_code = repaired_s[2] + repaired_s[3]
        # rto_start = 4
        # rto_end = 5
        # for i in range(len(repaired_s),rto_start,-1):
        #     if(repaired_s[i] in number):
        #         rto_end = i
        #         break

        # while(len(repaired_s) - rto_end + 1 > 4):
        #     rto_end+=1

        similarity_chart = {'0':['0','O',"Q","D"],"1":['1',"I","T","J"],"2":["2",'Z'],"3":["3"],"4":["4"],"5":["S"],"6":["C","G"],"7":["I","T"],"8":["B","E","R"], "9":["Q"], "A":["A","6"], "B":["B","8","R"], "C":["C","6","G"],"D":["D","0","O","Q"],"E":["E","8"],"F":["F","4"],"G":["H","6","N"],"H":["H","8","N"],"I":["I","1","J","T"],"J":["J","1","I","T"],"K":["K","8","R"],"L":["L","2"],"M":["M","8"],"N":["N","4","H"],"O":["O","0","D","Q"],"P":["P","9"],"Q":["Q","9"],"R":["R","8","K"],"S":["S","5"],"T":["T","1","I"],"U":["U","0","O","V"],"V":["V","1","Y"],"W":["W","4"],"X":["X","Y"],"Y":["Y","V"],"Z":["Z","5"]}

        if(np_st_code not in state_code):
            rectified_st_code = repaired_s[0]+repaired_s[1]
            for i in similarity_chart[repaired_s[0]]:
                for j in similarity_chart[repaired_s[1]]:
                    if(str(i+j) in state_code):
                        rectified_st_code = i+j
            repaired_s[0] = rectified_st_code[0]
            repaired_s[1] = rectified_st_code[1]
        # print(repaired_s)
        # if(np_st_code not in state_code):
        if(np_ditrict_code[0] not in number):
            # print(similarity_chart[repaired_s[2]][1])
            repaired_s[2] = str(similarity_chart[np_ditrict_code[0]][1])
        if(np_ditrict_code[1] not in number):
            # print(similarity_chart[repaired_s[3]][1])
            repaired_s[3] = str(similarity_chart[np_ditrict_code[1]][1])

        return "".join(repaired_s)
        
    # to rectify errors in type 2 license plate (old format)
    def repair_type_2(self,s):
        state_code = ["AP","AS","BR","BM","DL","GJ","HR","HP","JK","KA","KL","MD","MP","MH","MY","OR","PB","RJ","SK","TN","UP","WB"]
        number = ["0","1","2","3","4","5","6","7","8","9"]
        repaired_s = []
        for i in s:
            repaired_s.append(i)

        for i in range(2,len(s)):
            if(repaired_s[i] == 'O'):
                repaired_s[i] = "0"
            if(repaired_s[i] == "I"):
                repaired_s[i] = "1"
        np_st_code = repaired_s[0]+repaired_s[1]
        np_ditrict_code = repaired_s[2]+" "

        similarity_chart = {'0':['0','O',"Q","D"],"1":['1',"I","T","J"],"2":["2",'Z'],"3":["3"],"4":["4"],"5":["S"],"6":["C","G"],"7":["I","T"],"8":["B","E","R"], "9":["Q"], "A":["A","6"], "B":["B","8","R"], "C":["C","6","G"],"D":["D","0","O","Q"],"E":["E","8"],"F":["F","4"],"G":["H","6","N"],"H":["H","8","N"],"I":["I","1","J","T"],"J":["J","1","I","T"],"K":["K","8","R"],"L":["L","2"],"M":["M","8"],"N":["N","4","H"],"O":["O","0","D","Q"],"P":["P","9"],"Q":["Q","0"],"R":["R","8","K"],"S":["S","5"],"T":["T","1","I"],"U":["U","0","O","V"],"V":["V","1","Y"],"W":["W","4"],"X":["X","Y"],"Y":["Y","V"],"Z":["Z","2"]}

        if(np_st_code not in state_code):
            rectified_st_code = repaired_s[0]+repaired_s[1]
            for i in similarity_chart[repaired_s[0]]:
                for j in similarity_chart[repaired_s[1]]:
                    if(str(i+j) in state_code):
                        rectified_st_code = i+j
            repaired_s[0] = rectified_st_code[0]
            repaired_s[1] = rectified_st_code[1]
        if(np_ditrict_code[0] in number):
            # print(similarity_chart[repaired_s[2]][1])
            if(similarity_chart[np_ditrict_code[0]][1] not in number):
                repaired_s[2] = str(similarity_chart[np_ditrict_code[0]][1])
            else:
                repaired_s[2] = str(similarity_chart[np_ditrict_code[0]][2])

        for i in range(3,len(s)):
            if(repaired_s[i] not in number):
                repaired_s[i] = str(similarity_chart[repaired_s[i]][1])

        return "".join(repaired_s)
    
    def _predict(self,img_path):
        result1 = self.infer1(img_path)
        result2 = self.infer2(img_path)
        result3 = self.infer3(img_path)
        result1 = "".join(result1['predictions'][0]["rec_texts"])
        result2 = "".join(result2['predictions'][0]["rec_texts"])
        result3 = "".join(result3['predictions'][0]["rec_texts"])
        result1 = result1.upper()
        result2 = result2.upper()
        result3 = result3.upper()
        result1 = self.get_valid(result1)
        result2 = self.get_valid(result2)
        result3 = self.get_valid(result3)
        result = []
        print(result1,result2,result3)
        return [result1,result2,result3]

    def predict(self,img_path):
        self.clean_img(img_path)
        result = []
        result1 = ""
        result2 = ""
        result3 = ""
        
        for img in sorted(os.listdir(self.out_p1_path)):
            print(img)
            if(img.split(".")[-1] != "jpg"):
                continue
            pred = self._predict(os.path.join(self.out_p1_path,img))
#             pred_lp+=pred
            result1 += pred[0]
            result2 += pred[1]
            result3 += pred[2]
            os.remove(os.path.join(self.out_p1_path,img))
            
        # to remove improper license plates
        if(len(result1) <= 10):
            result.append(result1)
        if(len(result2) <= 10):
            result.append(result2)
        if(len(result3) <= 10):
            result.append(result3)
            
        for i in range(len(result)):
            typ = self.get_type(result[i])
            if(typ == 1):
                result[i] = self.repair_type_1(result[i])
            if(typ == 2):
                result[i] = self.repair_type_2(result[i])
        print(result)
        self.img_ori = None
        ### can work on further prunning the data###
        
        ###
        return result[0]
