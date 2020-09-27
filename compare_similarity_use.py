

"""
@author:  chih wei Wu
@contact: alanwu24@gmail.com
"""






#####
# import function
#####
import os
import sys
import cv2
import time
import math
import numpy as np

# RGB-> HSV
import colorsys


from PIL import Image

#####
# import pytorch
#####
import torch
import torch.nn as nn
import torch.nn.functional as F


#####
# import scipy
#####
from scipy import signal
from scipy.signal import convolve2d
import scipy.spatial.distance as dist






##########
# 比較位置:
##########
def compare_position(position_1, position_2,h_frame,w_frame):
    #print('position_1 size')
    #print(position_1.shape)
    # now_frame
    x_1 = position_1[0,0]
    y_1 = position_1[0,1]
    # before_frame
    x_2 = position_2[0,0]
    y_2 = position_2[0,1]
    
    ###
    # 中心距離差異
    ###
    center_dist = np.sqrt((x_1 - x_2) **2 + (y_1 - y_2) **2)
    #print('center_dist')
    #print(center_dist)
    ####
    # 計算跟exp(0)的差異
    ####
    # type1:
    #output_dist_similarity = math.exp(1/center_dist) - math.exp(0)
    # type2:
    Frame_diagonal_length = round( np.sqrt( h_frame**2 + w_frame **2)  )
    #print('Frame_diagonal_length')
    #print(Frame_diagonal_length)
    output_dist_similarity = center_dist/Frame_diagonal_length
    #print('output_dist_similarity')
    #print(output_dist_similarity)
    #final_dist_sim = math.exp(-output_dist_similarity)
    final_dist_sim = 1 - output_dist_similarity
    #if output_dist_similarity <= 100:
    #    final_dist_sim = output_dist_similarity/100
    #else:
    #    final_dist_sim = 1
    #print('final_dist_sim')
    #print(final_dist_sim)
    #print('\n')
    

    return final_dist_sim
#####
# 抓取dominate color of car's ROI
##### 

def get_dominant_color(image):
    # 顏色轉換成RGB
    #image = image.convert('RGB')
    image = Image.fromarray(image)
    image = image.convert('RGB')
    
    max_score = 0.0001
    dominant_color = None
    for count,(r,g,b) in image.getcolors(image.size[0]*image.size[1]):
        
        # 轉換成HSV --> 處理亮度問題:
        saturation = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]
        y = min(abs(r*2104+g*4130+b*802+4096+131072)>>13,235)
        y = (y-16.0)/(235-16)
        
        # 忽略太亮的顏色
        if y > 0.9:
            continue
        score = (saturation+0.1)*count
        if score > max_score:
            max_score = score
            dominant_color = (r,g,b)
    
    
    return dominant_color

# 計算圖片的MSE
###########
def RMSE(img1, img2):
        #squared_diff = (img1 -img2) ** 2
        #summed = np.sum(squared_diff)
        #num_pix = img1.shape[0] * img1.shape[1] * img1.shape[2]#img1 and 2 should have same shape
        #err = summed / num_pix
        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        err = err /  float(img1.shape[0] * img1.shape[1]* img1.shape[2])
        if err == 0:
            return err
        else:            
            ###取 sqrt --> RMSE
            return np.sqrt(err)
    
######
# PSNR:
######

def psnr_edge(edge_img1,edge_img2):
    
    img1 = edge_img1.astype(np.float64) /255
    img2 = edge_img2.astype(np.float64) /255
    mse = np.mean( (img1 - img2) **2 )
    if mse == 0:
        return 50
    else:
        pixel_max = 255
        output_psnr_edge = 10 * math.log10(pixel_max / math.sqrt(mse))
    
    return output_psnr_edge
    
################
# 計算圖片的餘弦距離
##################
def image_similarity_vectors_via_numpy(image1, image2):
    #image1 = get_thum(image1)
    #image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        #for pixel_tuple in image.getdata():
        vector.append(np.average(image))
        vectors.append(vector)
        # linalg=linear（線性）+algebra（代數），norm則表示範數
        # 求圖片的範數？？
        norms.append(np.linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是點積，對二維陣列（矩陣）進行計算
    res = np.dot(a / a_norm, b / b_norm)
    return res    
    

##########
# SSIM
##########

def cal_ssim(im1,im2):
    #assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    
    return ssim    
    

########
# CNN features
#######
class CMNet_Test(nn.Module):
    def __init__(self, input_channel=3, block_1=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )
        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        
        self.bn_prelu_1 = BNPReLU(32 + 3)
        
        
        # DAB Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.DAB_Block_1 = nn.Sequential()
        dilation_block_1 = [2, 2, 4, 4, 8, 8]
        
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=dilation_block_1[i]))
        
        self.bn_prelu_2 = BNPReLU(64)
        
        # type 1 : ok
        #self.classifier = nn.Sequential(Conv(64, input_channel, 3, 1, padding=1))
        # type 2:
        self.classifier = nn.Sequential(Conv(64, input_channel, 3, 1, padding=0))
    def forward(self, input):
        
        
        width = int(input.size()[2]/2)
        height = int(input.size()[3]/2)
        
        #output = self.conv_1(input)
        output0 = self.init_conv(input)
        
        # down-sample the image 1 times
        down_1 = self.down_1(input)
        
        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))
        
        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        
        output1 = self.bn_prelu_2(output1)
        
        out = self.classifier(output1)
        
        
        out = F.interpolate(out, (width,height), mode='bilinear', align_corners=False)
        
        return out



class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                             padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=True)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input



class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output

    
class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input    

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=True, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)           
            #self.bn_prelu = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output    
    
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output
    
########
# CNN features
#######    
    
    
    
#############
# 計算HSV 距離:
#############
def HSV_dist(image_1, image_2):
    
    # HSV_1:
    aH1, aS1, aV1 = cv2.split(image_1)    
    H1 = np.array(aH1).flatten()
    S1 = np.array(aS1).flatten()
    V1 = np.array(aV1).flatten()
        
    # HSV_2:
    aH2, aS2, aV2 = cv2.split(image_2)
    H2 = np.array(aH2).flatten()
    S2 = np.array(aS2).flatten()
    V2 = np.array(aV2).flatten()
    
    ####
    # 比較公式:
    ####
    R = 100.0
    angle = 30.0
    h = R * math.cos(angle / 180 * math.pi)
    r = R * math.sin(angle / 180 * math.pi)
    
    sum_result = 0.0
    for i in range(0, len(H1)):
        # HSV_1 calculate:
        x1 = r * V1[i] * S1[i] * math.cos(H1[i] / 180.0 * math.pi)
        y1 = r * V1[i] * S1[i] * math.sin(H1[i] / 180.0 * math.pi)
        z1 = h * (1 - V1[i])
        
        # HSV_2 calculate:
        x2 = r * V2[i] * S2[i] * math.cos(H2[i] / 180.0 * math.pi)
        y2 = r * V2[i] * S2[i] * math.sin(H2[i] / 180.0 * math.pi)
        z2 = h * (1 - V2[i])
        
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        
        sum_result = sum_result + dx * dx + dy * dy + dz * dz
        
    eucli_dean = np.sqrt(sum_result)
    
    print('eucli_dean')
    print(eucli_dean)
    
    return eucli_dean

##############
# SIFT key point compare:
##############

def sift_compare(image_1, image_2):
    
    #print('image_1 size')
    #print(image_1.shape)
    
    #######
    # 轉換成灰階:
    #######
    #image_1= cv2.cvtColor(image_1.astype('uint8'),cv2.COLOR_BGR2GRAY)
    #image_2= cv2.cvtColor(image_2.astype('uint8'),cv2.COLOR_BGR2GRAY)

    image_1= cv2.cvtColor(image_1.astype('uint8'),cv2.COLOR_BGR2RGB)
    image_2= cv2.cvtColor(image_2.astype('uint8'),cv2.COLOR_BGR2RGB)

    
    
    ## 創建 key points finder
    #finder = cv2.xfeatures2d.SIFT_create()
    finder = cv2.xfeatures2d.SURF_create()
    lowe_ratio = 0.75
    # find the keypoints and descriptors with SIFT
    kp1, des1 = finder.detectAndCompute(image_1,None)
    kp2, des2 = finder.detectAndCompute(image_2,None)
    
    #print('des2')
    #print(type(des2))
    #print(des2.shape)
    #print(type(des1))
    #print(type(des2))
    if des1 is  None or des2 is None:
        Score_SIFT = 0        
    else:    
        # BFMatcher with default params
        if des1.shape[0]  and des2.shape[0]> 0:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test
            good = []
            for i, pair in enumerate(matches):
                #for m,n in matches:
                    try:
                        m, n = pair
                        if m.distance < lowe_ratio*n.distance:
                            good.append([m])
                    except ValueError:
                        pass
            Score_SIFT = len(good) / (max(len(des1), len(des2)))
        else:
            Score_SIFT = 0
    #Score_SIFT = 1 - len(good) / (max(len(des1), len(des2)))
    #print('Score_SIFT')
    #print(Score_SIFT)
    
    return Score_SIFT


##############
# 主要顏色成分分數:
##############

def dominant_color_score(RGB_1, RGB_2):
    
    ## ex REID object:
    R_1 = RGB_1[0]
    G_1 = RGB_1[1]
    B_1 = RGB_1[2]
    
    ## now frame object
    R_2 = RGB_2[0]
    G_2 = RGB_2[1]
    B_2 = RGB_2[2]    
    
    ## R G B 分別差異
    
    R_gap = R_1 - R_2
    G_gap = G_1 - G_2
    B_gap = B_1 - B_2
    ## MSE 差異
    result_score =  np.sqrt ( (R_gap)**2 + (G_gap)**2 + (B_gap) **2) /3
    ## Mean L1 norm 差異
    #result_score = ( abs(R_gap) + abs(G_gap) + abs(B_gap)) /3
    
    
    return result_score


#########
# parital matching
########
def part_match(RGB_now, RGB_before):
    
    input_now = RGB_now
    input_compare = RGB_before
    #print('RGB_now.shape[1]')
    #print(RGB_now.shape[1])
    # 原本的 2020818
    #block_size = 10
    # 原本的 2020818 改
    block_size = 10
    
    all_block = (int(40/block_size))**2
    
    Part_similarity = 0
    
    for i_row in range(0, RGB_now.shape[0], block_size):
        for j_col in range(0,  RGB_now.shape[1], block_size):
            #print('j_col')
            #print(j_col)
            #start_point = j_col
            temp_now = input_now[i_row:i_row+block_size,j_col:j_col+block_size,:]
            temp_before = input_compare[i_row:i_row+block_size,j_col:j_col+block_size,:]
            
            output_RMSE = RMSE(temp_now,temp_before)
            
            #print('output_RMSE')
            #print(output_RMSE)
            if output_RMSE <=100:
                Part_RMSE_RGB = 1-(output_RMSE/100)
            else:
                Part_RMSE_RGB = 0
            
            Part_similarity = Part_similarity + Part_RMSE_RGB
            
    #print('=============')
    #print('Part_similarity')
    #print(Part_similarity/16)
    #print('\n')
            
    output_Part_similarity = Part_similarity/all_block
    
    #print('output_Part_similarity')
    #print(output_Part_similarity)
    
    return output_Part_similarity

def backproject(source, target, levels = 2, scale = 1):
        hsv = cv2.cvtColor(source,  cv2.COLOR_BGR2HSV)
        hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        # calculating object histogram
        # cv2.calcHist(影像, 通道, 遮罩, 區間數量, 數值範圍)
        top_0 = 180
        roihist = cv2.calcHist([hsv],[0, 1], None, \
            [levels, levels], [0, top_0, 0, 256] )
        #[0, 180, 0, 256]

        # normalize histogram and apply backprojection
        cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
        dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,top_0,0,256], scale)
        return dst

def largest_contours_rect(saliency):
        binary,contours, hierarchy = cv2.findContours(saliency * 1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key = cv2.contourArea)
        return cv2.boundingRect(contours[-1])


def refine_saliency_with_grabcut(img, saliency):
        rect = largest_contours_rect(saliency)
        bgdmodel = np.zeros((1, 65),np.float64)
        fgdmodel = np.zeros((1, 65),np.float64)
        saliency[np.where(saliency > 0)] = cv2.GC_FGD
        mask = saliency
        # cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_RECT
        cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, \
                    1, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        return mask


def compare_salient_part(image1, image2):
    
    #print('image1')
    #print(image1.shape)
    #print('image2')
    #print(image2.shape)
    image1= cv2.cvtColor(image1.astype('uint8'),cv2.COLOR_BGR2RGB)
    image2= cv2.cvtColor(image2.astype('uint8'),cv2.COLOR_BGR2RGB)
    
    
    ### pyrMeanShiftFiltering 順序: 原始image, 物理空間半徑, 色彩空間半徑, 輸出圖像, 金字塔層數
    sp = 10
    sr = 20
    ## 2020818
    #image1 = cv2.resize(image1, dsize=(100, 100))
    #image2 = cv2.resize(image2, dsize=(100, 100))
    ## mean shift
    image1_mean_shift_result = cv2.pyrMeanShiftFiltering(image1, sp, sr, image1, 2)
    image2_mean_shift_result = cv2.pyrMeanShiftFiltering(image2, sp, sr, image2, 2)
    ## backproject 2020818 因為 image size沒有 resize成 100 太慢了
    #image1_backproj = np.uint8(backproject(image1_mean_shift_result, image1_mean_shift_result, levels = 16))
    #image2_backproj = np.uint8(backproject(image2_mean_shift_result, image2_mean_shift_result, levels = 16))
    image1_backproj = np.uint8(backproject(image1_mean_shift_result, image1_mean_shift_result, levels = 8))
    image2_backproj = np.uint8(backproject(image2_mean_shift_result, image2_mean_shift_result, levels = 8))
    ## normalize
    cv2.normalize(image1_backproj,image1_backproj,0,255,cv2.NORM_MINMAX)
    cv2.normalize(image2_backproj,image2_backproj,0,255,cv2.NORM_MINMAX)
    ## find saliency part
    image1_saliencies = [image1_backproj, image1_backproj, image1_backproj]
    image1_saliency = cv2.merge(image1_saliencies)
    image2_saliencies = [image2_backproj, image2_backproj, image2_backproj]
    image2_saliency = cv2.merge(image2_saliencies)   
    ## mean sfit 2
    sp_2 = 10
    sr_2 = 20
    cv2.pyrMeanShiftFiltering(image1_saliency, sp_2, sr_2, image1_saliency, 2)
    cv2.pyrMeanShiftFiltering(image2_saliency, sp_2, sr_2, image2_saliency, 2)
    ## convert to gray level
    image1_saliency = cv2.cvtColor(image1_saliency, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(image1_saliency, image1_saliency)
    
    image2_saliency = cv2.cvtColor(image2_saliency, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(image2_saliency, image2_saliency)    
    
    # (T, saliency) = cv2.threshold(saliency, 200, 255, cv2.THRESH_BINARY)
    (image1_T, image1_saliency) = cv2.threshold(image1_saliency, 200, 255, cv2.THRESH_BINARY)
    (image1_T, image2_saliency) = cv2.threshold(image2_saliency, 200, 255, cv2.THRESH_BINARY)
    
    ## find the saliency part contour
    image1_output_mask = refine_saliency_with_grabcut(image1,image1_saliency)
    image2_output_mask = refine_saliency_with_grabcut(image2,image2_saliency)
    #######
    # 產生最後圖片-1:
    #######
    image_1_final_saliency = image1*image1_output_mask[:,:,np.newaxis]
    image_2_final_saliency = image1*image2_output_mask[:,:,np.newaxis]
    
    ######
    # 先resize 20200818
    ######
    #image_1_final_saliency = cv2.resize(image_1_final_saliency, dsize=(40, 40))
    #image_2_final_saliency = cv2.resize(image_2_final_saliency, dsize=(40, 40))
    ######
    # 比較 MSE
    ######
    
    #print('image_1_final_saliency')
    #print(image_1_final_saliency.shape)
    #print('image_2_final_saliency')
    #print(image_2_final_saliency.shape)
    output_saliency_RMSE = RMSE(image_1_final_saliency,image_2_final_saliency)
    #print('output_saliency_RMSE')
    #print(output_saliency_RMSE)
    if output_saliency_RMSE <=100:
        saliency_RMSE_RGB = 1-(output_saliency_RMSE/100)
    else:
        saliency_RMSE_RGB = 0    
    #print('saliency_RMSE_RGB')
    #print(saliency_RMSE_RGB)    
    return saliency_RMSE_RGB

    
#if __name__ == '__main__':
    #model = CMNet_Test(input_channel=3)
    #model = CMNet_Test(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=11, zoom_factor=1, use_ppm=True, pretrained=True)
    #x = torch.rand(1, 3, 40, 40)
    #y = model(x)
    #print('shape')
    #print(y.shape)