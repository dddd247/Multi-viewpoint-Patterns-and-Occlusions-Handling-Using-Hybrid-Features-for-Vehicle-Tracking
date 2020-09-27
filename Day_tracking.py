



"""
@author:  chih wei Wu
@contact: alanwu24@gmail.com
"""




import os
import sys
import cv2
import time
import math
import numpy as np
import colorsys
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.signal import convolve2d
import scipy.spatial.distance as dist
from compare_similarity_use import *


######################
#
#   main program
#
#########################

########
# enter your output path
########
crop_outpuut_save_path = './final_real_world_detect_output/Test/Day_normal/Day_normal_detect_output/'
################
# ROI size change 
# exclude too small area  
##############
ROI_size_change =200


######################
# enter you labels on each video frame
######################
array_compare = np.array([(0,1,2,3,5,6,7),   # frame 1 , 7 objects
                          (0,1,2,3,4,5,6,7),   # frame 2 , 8 objects
                          (0,1,2,3,4,5,6,7),   # frame 3 , 8 objects
                          (0,1,2,3,5,6,7),   # frame 4 , 7 objects
                          (0,1,2,3,5,6,7),   # frame 5 , 7 objects
                          (0,1,2,3,5,6,7),   # frame 6 , 8 objects
                          (0,1,2,3,5,6,8,7),   # frame 7 , 8 objects
                          (0,1,2,3,5,6,7),   # frame 8 , 7 objects 
                          (0,1,2,3,4,5,6,7),   # frame 9 , 8 objects
                          (0,1,2,3,5,6,7),   # frame 10 , 7 objects
                          (0,1,2,3,5,6,7),   # frame 11 , 7 objects
                          (0,1,2,3,4,5,6,7),   # frame 12 , 8 objects
                          (0,1,2,3,4,5,6,7),   # frame 13 , 8 objects
                          (0,1,2,3,4,5,6,7),   # frame 14 , 8 objects
                          (0,1,2,3,4,5,6,7),  # frame 15 , 8 objects
                          (0,1,2,3,4,5,6,7),   # frame 16 , 8 objects
                          (0,1,2,3,5,6,7),   # frame 17 , 7 objects
                          (0,1,2,3,5,6,8,7),   # frame 18 , 8 objects
                          (0,1,2,3,5,6,8,7),   # frame 19 , 8 objects
                          (0,1,2,3,4,5,6,8,7),   # frame 20 , 9 objects

                          ])


############################################################
# save the result of current frame with previous label
############################################################
list_now_frame = []

############################################################
# save the result of current frame with new label
####### ####################################################
list_now_frame_new = []

############################################################
# save the cropped vehicles' features
############################################################
crop_history_image = np.zeros([40,100,3,20000])

############################################################
# calculate the number of video frames
############################################################
frame_counter = 0

############################################################
# frame 0 counter
############################################################
frame_0_counter = 0


############################################################
# save Re-ID result -- image
############################################################

############################################################
# save the Re-Id obejcts:
############################################################
compare_REID_number = 0
############################################################
# calculate the number of vehicles:
############################################################
all_cars_count = 0


############################################################
#  Object detector: yolov3
############################################################

CONFIDENCE = 0.6
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "cfg/yolov3.cfg"
weights_path = "weights/yolov3.weights"
font_scale = 1
thickness = 2
labels = open("data/coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]



############################################################
# calculate the number of comparison pair
#  之前的ID需要比較的部分:
############################################################
compare_number = 0



############################################################
# save the comparison pair
############################################################

compare_pair_save_path = './final_real_world_detect_output/Test/Day_normal/Day_normal_read_world_first_two_frame_compair_pair/'


############################################################
# save each id image:
############################################################
id_save_path = './final_real_world_detect_output/Test/Day_normal/Day_normal_each_id_sace/'


############################################################
# calculate the number of comparison vehicles
############################################################
compare_count = 0
############################################################
# computation time:
############################################################
time_all = 0
############################################################
# the number of correct Re-ID:
############################################################
counter_re_id = 0
############################################################
# the number of all vheilces:
############################################################
counter_all_cars = 0



############################################################
# image folder input path
############################################################
image_folder_path = './final_real_world_detect_output/Test/Day_normal/Day_normal_original_video_output/'


############################################################
# stop frame
############################################################
stop_frame = 21

while frame_counter <stop_frame:
    print('\n')
    print('Frame !!!!!!!!!!!!!!!!!!!')
    print("Frame %d" %(frame_counter))
    print('\n')

    image_Name = str(frame_counter+1)+".jpg"
    image = cv2.imread(os.path.join(image_folder_path, image_Name))
    
    
    h, w = image.shape[:2]
    h_frame = h
    w_frame = w
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print("Time took on detect:", time_took)
    boxes, confidences, class_ids = [], [], []
    boxes_others, confidences_others, class_ids_others = [], [], []  
    
    start = time.perf_counter()
    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:] 
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                if class_id == 2 and  y <= 400: 
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                else:
                    boxes_others.append([x, y, int(width), int(height)])
                    confidences_others.append(float(confidence))
                    class_ids_others.append(class_id)
    ##############3            
    # perform the non maximum suppression given the scores defined before        
    ###############
    ## for cars
    idxs_Cars = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    ## for other objects
    idxs_others = cv2.dnn.NMSBoxes(boxes_others,confidences_others,SCORE_THRESHOLD,IOU_THRESHOLD)
    ###############
    # save cropped image:!
    # 1. cars
    # 2. other objects
    ###############
    ## for cars
    Cars_save_picture_now = np.zeros([40,40,3,len(idxs_Cars)])
    ## for other objects
    Others_save_picture_noew = np.zeros([40,40,3,len(idxs_others)])
    
    font_scale = 0.5
    thickness = 2

    thumb_w=40 
    thumb_h=40
    off_x=10
    off_y=10
    i_counter = 1
    new_counter = 1
    
    id_number  = 0
    print('the number of detected vehicles: %d' %(len(idxs_Cars)))
    # ensure at least one detection exists
    
    
    
    ##################
    # 觀察 block size of car object
    ##################
    if len(idxs_Cars) > 0:
        counter_large_car_counter = 0
        
        idxs_cars_new = np.zeros([1,1])
        check_point_counter = 0
        list_conter = []
        list_compare = []
        for check_size in idxs_Cars.flatten():
            # extract the bounding box coordinates
            x, y = boxes[check_size][0], boxes[check_size][1] 
            w, h = boxes[check_size][2], boxes[check_size][3]
            
            check_size_ROI = w * h
 
                    
            if check_size_ROI > ROI_size_change:
                list_compare.append(check_size)
                list_conter.append(check_size)
            
        for i_rerange in range(0, len(list_compare),1):
             compare_list = list_compare
             for j_compare in range( 0, len(compare_list),1):
                 if boxes[list_compare[i_rerange]][0] < boxes[compare_list[j_compare]][0]:
                     temp_1 = compare_list[j_compare]
                     temp_2 = list_compare[i_rerange]
                     compare_list[j_compare] = temp_2
                     list_compare[i_rerange] = temp_1
                 else:
                     list_compare[i_rerange] = list_compare[i_rerange] 
                
    else:
        list_conter = []
    
    list_conter = list_compare
    
    if frame_counter == 0 and frame_0_counter == 0:
    

        Reid_features = np.zeros([40,100,3,len(list_conter)])

        ReID_object = np.zeros([40,40,3,len(list_conter)])

        Reid_position = np.zeros([1,2,len(list_conter)])

        
        frame_0_counter = frame_0_counter + 1
        
        frame_0_car_counter = 0
    
    if frame_counter > 0:
        counter_all_cars = counter_all_cars +len(list_conter)
    
        

    if len(list_conter) > 0:

        cars_id_number = 0
        list_now_frame = []
        list_now_frame_new = []
        for i in list_conter:
            
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            
            ROI_size =  w * h
            
            if frame_counter == 0:
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)                
                
                
                
                text = f"{labels[class_ids[i]]}-{frame_0_car_counter}: {confidences[i]:.2f}"
                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                # add opacity (transparency to the box)
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                # now put the text (label: confidence %)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,\
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
                id_number = id_number + 1
                
                
                frame_0_car_counter = frame_0_car_counter + 1
                detect_x_place = 80
                cv2.putText(image, 'Detected Vehicles', (detect_x_place,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
                
                if boxes[i][0] <=0:
                    boxes[i][0] = 1
                


                if len(idxs_others) > 0:

                    for j in idxs_others.flatten():

                        x_cars, y_cars = boxes[i][0], boxes[i][1]
                        w_cars, h_cars = boxes[i][2], boxes[i][3]
                        ## other objects
                        x_others, y_others = boxes_others[j][0], boxes_others[j][1]
                        w_others, h_others = boxes_others[j][2], boxes_others[j][3]

                        if (x_cars < x_others) and (y_cars < y_others ) and\
                            ( (x_cars+w_cars) > (x_others+w_others) ) and\
                            ( (y_cars+h_cars) > (y_others+h_others)):
  
                                before_crop = image.copy()

                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others] = 0

                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others])

                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number

                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                                # R
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]

                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]

                        elif ( x_cars < x_others ) and ( (x_cars + w_cars) > ( x_others +w_others)) and\
                            (y_others > y_cars) and ( (y_others + h_others) > (y_cars + h_cars)):

                                before_crop = image.copy()

                                before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes_others[j][0]+w_others] = 0

                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes_others[j][0]+w_others])

                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number

                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])

                                # R
                                before_crop[boxes_others[j][1][0]:boxes[i][1][0]+h_cars, boxes_others[j][0][0]:boxes_others[j][0][0]+w_others,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes_others[j][1][1]:boxes[i][1][1]+h_cars, boxes_others[j][0][1]:boxes_others[j][0][1]+w_others,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes_others[j][1][2]:boxes[i][1][2]+h_cars, boxes_others[j][0][2]:boxes_others[j][0][2]+w_others,2] = Dominant_RGB_value[2]

                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]

                        elif( x_cars < x_others ) and ( (x_cars + w_cars) > ( x_others +w_others)) and\
                            (y_others < y_cars) and ( (y_others + h_others) < (y_cars + h_cars)):

                                before_crop = image.copy()

                                before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others] = 0

                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others])
         
                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number

                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])

                                # R
                                before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]
                                ## 儲存處理過後的結果 --> 將car's bounding 處理過後的結果儲存下來!:
                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
                        

                        elif ( x_cars > x_others )and ( (x_cars + w_cars) > ( x_others + w_others) ) and \
                        ( y_cars < y_others ) and ( (y_cars + h_cars) > (y_others+h_others) ):
                               
                            

                                before_crop = image.copy()

                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others] = 0

                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others])

                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number

                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])

                                # R
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]

                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]

                        elif( x_cars < x_others )and ( (x_cars + w_cars) < ( x_others + w_others) ) and \
                        ( y_cars < y_others ) and ( (y_cars + h_cars) > (y_others+h_others) ):

                                before_crop = image.copy()

                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars] = 0

                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars])
                           
                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number

              
                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])

                                # R
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,2] = Dominant_RGB_value[2]

                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
   
                        elif ( x_cars < x_others ) and ( (x_cars + w_cars) < (x_others + w_others) ) and\
                            ( y_cars > y_others) and ( (y_cars + h_cars) > (y_others + h_others)):                           

                            before_crop = image.copy()
         
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars] = 0
   
                            div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars])
       
                            mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
         
                            Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                  
                            # R
                            #print(Dominant_RGB_value[0])
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,0] = Dominant_RGB_value[0]
                            # G
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,1] = Dominant_RGB_value[1]
                            # B
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,2] = Dominant_RGB_value[2]
         
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]  
            
                        elif ( x_cars < x_others ) and ( (x_cars + w_cars) < (x_others + w_others) ) and\
                            ( y_cars < y_others) and ( (y_cars + h_cars) < (y_others + h_others)):
        
                            before_crop = image.copy()
                      
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars] = 0
                     
                            div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars])

                            mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
              
                            Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])

                            # R
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars,0] = Dominant_RGB_value[0]
                            # G
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars,1] = Dominant_RGB_value[1]
                            # B
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars,2] = Dominant_RGB_value[2]
        
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars] 
           
                        elif ( x_cars > x_others ) and ( (x_cars + w_cars) > (x_others + w_others) ) and\
                            ( y_cars > y_others) and ( (y_cars + h_cars) > (y_others + h_others)):
           
                            before_crop = image.copy()
                       
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others] = 0
                           
                            div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others])                      
                           
                            mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                    
                            Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                      
                            # R
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                            # G
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                            # B
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]
                   
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
                
                        elif( x_cars > x_others ) and ( (x_cars + w_cars) > (x_others + w_others) ) and\
                            ( y_cars < y_others) and ( (y_cars + h_cars) < (y_others + h_others)):
                       
                            before_crop = image.copy()
                
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others] = 0
                                                    
                            div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others])                      
                           
                            mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                    
                   
                            Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                       
                          
                            # R
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                            # G
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                            # B
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]
            
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]                        
                        
                        else:
                            before_crop = image.copy()
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
                            
                else:
                    before_crop = image.copy()
                    thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h,boxes[i][0]:boxes[i][0]+w]
                
         
                ### resize output :
                vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))
     
                start_x = 0
                start_x_Stop = start_x + i_counter*thumb_w+(i_counter+1) * off_x
             
                if (start_x_Stop+ thumb_w) <= image.shape[1]:
                    image[off_y + 40:off_y + thumb_h + 40, start_x_Stop:start_x_Stop + thumb_w, :] = vehicle_thumb
                    
                else:
                    start_x = 0
                    start_x_Stop_new = start_x + new_counter*thumb_w+(new_counter+1) * off_x
                    image[off_y + 90:off_y + thumb_h + 90, start_x_Stop_new:start_x_Stop_new + thumb_w, :] = vehicle_thumb
                    new_counter = new_counter + 1
                        
                ### save output cars 
        
                print(' cars_id_number  count!!')
                print(cars_id_number)
                
  
                Reid_position[0,0,cars_id_number] = boxes[i][0] + round(w/2)
                Reid_position[0,1,cars_id_number] = boxes[i][1] + round(h/2)
             
                
                
                
                
                #########
                # 1.HSV features
                ########
                Cars_save_picture_now[:40,:40,:,cars_id_number] = vehicle_thumb[:,:,:]

                
         
                temp_image = Cars_save_picture_now[:40,:40,:,cars_id_number]
                ### cars_id_number

                Reid_features[:40,:40,:,cars_id_number] = cv2.cvtColor(temp_image.astype(np.uint8) ,cv2.COLOR_RGB2HSV)
                ############
                # 2. CNN features
                ############

                model = CMNet_Test(input_channel=3).double()
                Tensor_temp_image =np.zeros([40,40,3,1])
                Tensor_temp_image[:,:,:,0] = temp_image[:,:,:]
                Conv_output_feature = model((torch.from_numpy(Tensor_temp_image)).permute(3,2,0,1))
                Conv_output_feature = Conv_output_feature.permute(2,3,1,0)
                Reid_features[:20,40:60,:,cars_id_number] = Conv_output_feature.detach().numpy()[:,:,:,-1]
                ############
                # 3. edge features
                ############
                ##########
                # edge feature:
                ##########
                #######
                # Edge1:canny_edge
                ######
                Resize_temp_image = cv2.resize(temp_image, (20, 20), interpolation=cv2.INTER_CUBIC)
                Resize_temp_image_Gray = cv2.cvtColor(Resize_temp_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                Edge_feature_temp_image_canny = cv2.Canny(Resize_temp_image_Gray, 30, 150)
                #######
                # Edge2 and 3: sobel_x and sobel_y
                ######
                sobelx = cv2.Sobel(Resize_temp_image_Gray, cv2.CV_64F, 1, 0)
                sobely = cv2.Sobel(Resize_temp_image_Gray, cv2.CV_64F, 0, 1)
                sobelx = np.uint8(np.absolute(sobelx))
                sobely = np.uint8(np.absolute(sobely))
 
            
                Reid_features[20:40,40:60,0,cars_id_number] = Edge_feature_temp_image_canny
                Reid_features[20:40,40:60,1,cars_id_number] = sobelx
                Reid_features[20:40,40:60,2,cars_id_number] = sobely
                ############
                # 4. RGB feature
                ############
         
                Reid_features[:40,60:100,:,cars_id_number] = temp_image

                
                
                #############
                # save REID cropped images:
                #############
                
     
                ReID_object[:,:,:,cars_id_number] = Cars_save_picture_now[:40,:40,:,cars_id_number]
                ############
                # frame 1 : reid features map
                ############

                i_counter = i_counter +1

                all_cars_count = all_cars_count + len(idxs_Cars)
            ##########################
            # after 0 frame :
            ##########################
              
            else:    
                
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)  

                temp_Reid_features = np.zeros([40,100,3,len(list_conter)])
 
                temp_Reid_position = np.zeros([1,2,len(list_conter)])                

                detect_x_place = 80
                cv2.putText(image, 'Detected Vehicles', (detect_x_place,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
      
                if boxes[i][0] <=0:
                    boxes[i][0] = 1
  
                if len(idxs_others) > 0:
                    
                    for j in idxs_others.flatten():
                       
                        # extract the bounding box coordinates 
                        ## cars
                        x_cars, y_cars = boxes[i][0], boxes[i][1]
                        w_cars, h_cars = boxes[i][2], boxes[i][3]
                        ## other objects
                        x_others, y_others = boxes_others[j][0], boxes_others[j][1]
                        w_others, h_others = boxes_others[j][2], boxes_others[j][3]
                     
                        
                    
                        if (x_cars < x_others) and (y_cars < y_others ) and\
                            ( (x_cars+w_cars) > (x_others+w_others) ) and\
                            ( (y_cars+h_cars) > (y_others+h_others)):
                              
                                before_crop = image.copy()
                              
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others] = 0
                              
                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others])
                               
                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                               
                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                          
                                # R
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]
                                
                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
                      
                        elif ( x_cars < x_others ) and ( (x_cars + w_cars) > ( x_others +w_others)) and\
                            (y_others > y_cars) and ( (y_others + h_others) > (y_cars + h_cars)):
                               
                                before_crop = image.copy()
                               
                                before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes_others[j][0]+w_others] = 0
                              
                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes_others[j][0]+w_others])
                               
                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                              
                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                              
                                # R
                                before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]
                               
                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
               
                        elif( x_cars < x_others ) and ( (x_cars + w_cars) > ( x_others +w_others)) and\
                            (y_others < y_cars) and ( (y_others + h_others) < (y_cars + h_cars)):
                             
                                before_crop = image.copy()
                              
                                before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others] = 0
                            
                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others])
                                
                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                            
                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                             
                                # R
                                before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]
                                
                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
                        
                     
                        elif ( x_cars > x_others )and ( (x_cars + w_cars) > ( x_others + w_others) ) and \
                        ( y_cars < y_others ) and ( (y_cars + h_cars) > (y_others+h_others) ):
                               
                            
                            
                                before_crop = image.copy()
                              
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others] = 0
                               
                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others])
                              
                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                              
                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                               
                                # R
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]
                           
                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
                  
                        elif( x_cars < x_others )and ( (x_cars + w_cars) < ( x_others + w_others) ) and \
                        ( y_cars < y_others ) and ( (y_cars + h_cars) > (y_others+h_others) ):
                              
                                before_crop = image.copy()
                                   
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars] = 0
                                
                                div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars])
                                                              
                                mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                              
                                Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                             
                                # R
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,0] = Dominant_RGB_value[0]
                                # G
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,1] = Dominant_RGB_value[1]
                                # B
                                before_crop[boxes_others[j][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,2] = Dominant_RGB_value[2]
                                
                                thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
                     
                        elif ( x_cars < x_others ) and ( (x_cars + w_cars) < (x_others + w_others) ) and\
                            ( y_cars > y_others) and ( (y_cars + h_cars) > (y_others + h_others)):                           
                            
                            before_crop = image.copy()
                               
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars] = 0
                          
                            div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars])
                          
                            mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                           
                            Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                         
                            # R
                            #print(Dominant_RGB_value[0])
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,0] = Dominant_RGB_value[0]
                            # G
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,1] = Dominant_RGB_value[1]
                            # B
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes_others[j][0]:boxes[i][0]+w_cars,2] = Dominant_RGB_value[2]
    
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]  
                        # 右下
                        elif ( x_cars < x_others ) and ( (x_cars + w_cars) < (x_others + w_others) ) and\
                            ( y_cars < y_others) and ( (y_cars + h_cars) < (y_others + h_others)):
         
                            before_crop = image.copy()
                              
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars] = 0
                      
                            div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars])
                          
                            mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                  
                            Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
             
                            # R
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars,0] = Dominant_RGB_value[0]
                            # G
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars,1] = Dominant_RGB_value[1]
                            # B
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes_others[j][0]:boxes[i][0]+w_cars,2] = Dominant_RGB_value[2]
                           
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars] 
                    
                        elif ( x_cars > x_others ) and ( (x_cars + w_cars) > (x_others + w_others) ) and\
                            ( y_cars > y_others) and ( (y_cars + h_cars) > (y_others + h_others)):
                         
                            before_crop = image.copy()
                                 
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others] = 0
                            
                            div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others])                      
                       
                            mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                          
                            Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                 
                            # R
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                            # G
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                            # B
                            before_crop[boxes[i][1]:boxes_others[j][1]+h_others, boxes[i][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]
                      
                 
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
                   
                        elif( x_cars > x_others ) and ( (x_cars + w_cars) > (x_others + w_others) ) and\
                            ( y_cars < y_others) and ( (y_cars + h_cars) < (y_others + h_others)):
                  
                            before_crop = image.copy()
                          
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others] = 0
                                                 
                            div_number = np.size(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) - \
                                    np.size(before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others])                      
                           
                            mean_add = np.sum(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars]) / div_number
                           
                            Dominant_RGB_value = get_dominant_color(before_crop[boxes[i][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes[i][0]+w_cars])
                    
                            # R
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others,0] = Dominant_RGB_value[0]
                            # G
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others,1] = Dominant_RGB_value[1]
                            # B
                            before_crop[boxes_others[j][1]:boxes[i][1]+h_cars, boxes[i][0]:boxes_others[j][0]+w_others,2] = Dominant_RGB_value[2]
                     
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]                        
                        
                        else:
                            before_crop = image.copy()
                            thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h_cars,boxes[i][0]:boxes[i][0]+w_cars]
            
                else:
                    before_crop = image.copy()
                    thumbnail = before_crop[boxes[i][1]:boxes[i][1]+h,boxes[i][0]:boxes[i][0]+w]


                
       
                vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))

                start_x = 0
                start_x_Stop = start_x + i_counter*thumb_w+(i_counter+1) * off_x
                
 
                if (start_x_Stop+ thumb_w) <= image.shape[1]:
                    image[off_y + 40:off_y + thumb_h + 40, start_x_Stop:start_x_Stop + thumb_w, :] = vehicle_thumb
                    
                else:
                    start_x = 0
                    start_x_Stop_new = start_x + new_counter*thumb_w+(new_counter+1) * off_x
                    image[off_y + 90:off_y + thumb_h + 90, start_x_Stop_new:start_x_Stop_new + thumb_w, :] = vehicle_thumb
                    new_counter = new_counter + 1
                        
      
                Cars_save_picture_now[:40,:40,:,cars_id_number] = vehicle_thumb[:,:,:]
     
                temp_Reid_position[0,0,cars_id_number] = boxes[i][0] + round(boxes[i][2])
                temp_Reid_position[0,1,cars_id_number] = boxes[i][1] + round(boxes[i][3])                
                ##########
                # 抽取特徵:
                # 1. hsv
                # 2. CNN
                # 3. edge
                # 4. RGB
                #########
                # 1.HSV features
                ########

                temp_image = Cars_save_picture_now[:40,:40,:,cars_id_number]
                temp_Reid_features[:40,:40,:,cars_id_number] = cv2.cvtColor(temp_image.astype(np.uint8) ,cv2.COLOR_RGB2HSV)
                ############
                # 2. CNN features
                ############
     
                model = CMNet_Test(input_channel=3).double()
                Tensor_temp_image =np.zeros([40,40,3,1])
                Tensor_temp_image[:,:,:,0] = temp_image[:,:,:]
                Conv_output_feature = model((torch.from_numpy(Tensor_temp_image)).permute(3,2,0,1))
                Conv_output_feature = Conv_output_feature.permute(2,3,1,0)
  
                temp_Reid_features[:20,40:60,:,cars_id_number] = Conv_output_feature.detach().numpy()[:,:,:,-1]
                ############
                # 3. edge features
                ############

                #######
                # Edge1:canny_edge
                ######                
                Resize_temp_image = cv2.resize(temp_image, (20, 20), interpolation=cv2.INTER_CUBIC)
                Resize_temp_image_Gray = cv2.cvtColor(Resize_temp_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                Edge_feature_temp_image_canny = cv2.Canny(Resize_temp_image_Gray, 30, 150)
                #######
                # Edge2 and 3: sobel_x and sobel_y
                ######
                sobelx = cv2.Sobel(Resize_temp_image_Gray, cv2.CV_64F, 1, 0)
                sobely = cv2.Sobel(Resize_temp_image_Gray, cv2.CV_64F, 0, 1)
                sobelx = np.uint8(np.absolute(sobelx))
                sobely = np.uint8(np.absolute(sobely))

                temp_Reid_features[20:40,40:60,0,cars_id_number] = Edge_feature_temp_image_canny
                temp_Reid_features[20:40,40:60,1,cars_id_number] = sobelx
                temp_Reid_features[20:40,40:60,2,cars_id_number] = sobely
                ############
                # 4. RGB feature
                ############ 
                temp_Reid_features[:40,60:100,:,cars_id_number] = temp_image

                Best_Score_number = 0

                
                for i_comp in range(0,compare_REID_number):   

                    position_now = np.zeros([1,2])
                    position_before_f = np.zeros([1,2])

                    position_now = temp_Reid_position[:,:,cars_id_number]
                    position_before_f = Reid_position[:,:,i_comp]

                    Score_position = compare_position(position_now,position_before_f,h_frame,w_frame)

                    MSE_RGB_Temp_now_frame = temp_Reid_features[:40,60:100,:,cars_id_number]

                    MSE_RGB_ex_feature = Reid_features[:40,60:100,:,i_comp]
                 
                    
           
                    Compare_RMSE_RGB = RMSE(MSE_RGB_ex_feature,MSE_RGB_Temp_now_frame)
                    

                    HSV_Temp_now_frame = temp_Reid_features[:40,:40,:,cars_id_number]

                    HSV_ex_feature = Reid_features[:40,:40,:,i_comp]                    
     
                    cor_HSV =  np.corrcoef(HSV_ex_feature.flat, HSV_Temp_now_frame.flat)
                    cor_HSV_Result = cor_HSV[0, 1]
               
                    
                    
 
                    Conv_Temp_now_frame = temp_Reid_features[:20,40:60,:,cars_id_number]
           
                    Conv_ex_feature = Reid_features[:20,40:60,:,i_comp]
     
                    Conv_ex_feature_flat = Conv_ex_feature.flatten()
                    Conv_Temp_now_frame_flat = Conv_Temp_now_frame.flatten()
                   
                    Conv_dot = np.dot(Conv_ex_feature_flat, Conv_Temp_now_frame_flat)
                    norm_Conv_ex_feature_flat = np.linalg.norm(Conv_ex_feature_flat)
                    norm_compare_Conv_Temp_flat = np.linalg.norm(Conv_Temp_now_frame_flat)
                    cos_Cov_result = Conv_dot / (norm_compare_Conv_Temp_flat * norm_Conv_ex_feature_flat)

          
                    Edge_Canny_temp_now_frame = temp_Reid_features[20:40,40:60,0,cars_id_number]
                    Edge_Sobelx_temp_now_frame = temp_Reid_features[20:40,40:60,1,cars_id_number]
                    Edge_Sobely_temp_now_frame = temp_Reid_features[20:40,40:60,2,cars_id_number]                    
             
                    Edge_Canny_ex_feature = Reid_features[20:40,40:60,0,i_comp]
                    Edge_Sobelx_ex_feature = Reid_features[20:40,40:60,1,i_comp]
                    Edge_Sobely_ex_feature = Reid_features[20:40,40:60,2,i_comp] 
                    
                    
               
                    # canny
                    SSIM_result_Canny_edge = cal_ssim(Edge_Canny_ex_feature,Edge_Canny_temp_now_frame)
                    # sobel_x
                    SSIM_result_Sobelx_edge = cal_ssim(Edge_Sobelx_ex_feature,Edge_Sobelx_temp_now_frame)
                    # sobel_y
                    SSIM_result_Sobely_edge = cal_ssim(Edge_Sobely_ex_feature,Edge_Sobely_temp_now_frame)
                    
                    
                    edge_all_Score = 0.5*SSIM_result_Canny_edge + 0.25*SSIM_result_Sobelx_edge + 0.25*SSIM_result_Sobely_edge
                    
                    ###########
                    # compare "Edge" PSNR
                    ###########
                    output_canny_psnr=psnr_edge(Edge_Canny_ex_feature,Edge_Canny_temp_now_frame)
     
                    if Compare_RMSE_RGB <=100:
                        RMSE_RGB_compare_Score = 1-(Compare_RMSE_RGB/100)
                    else:
                        RMSE_RGB_compare_Score = 0
                    
                    
                    
          
                    Key_point_compare_score = sift_compare(MSE_RGB_ex_feature,MSE_RGB_Temp_now_frame)
                    
                    
                    
               
                    Ex_feature_dominant_color_result = get_dominant_color((MSE_RGB_ex_feature*255).astype(np.uint8))                    
                    Now_frame_dominant_color_result = get_dominant_color((MSE_RGB_Temp_now_frame*255).astype(np.uint8)) 
                    
                    Dominant_color_dist = dominant_color_score(Ex_feature_dominant_color_result,Now_frame_dominant_color_result)
               
                    
                    if Dominant_color_dist <=100:
                        Dominant_color_compare_score = 1 - ((Dominant_color_dist)/100)
                    else:
                        Dominant_color_compare_score = 0
            
                    
                    RGB_temp_now = MSE_RGB_Temp_now_frame
                    RGB_before = MSE_RGB_ex_feature
                    Part_Score = part_match(RGB_temp_now,RGB_before)
                    
                
                    Saliency_Score = compare_salient_part(MSE_RGB_Temp_now_frame,MSE_RGB_ex_feature)
                    
                    ####
                    # global feature, partial similarity, salient section, position correlation
                    ####
                    
                    Score_number_G = 0.15 * Key_point_compare_score + 0.375* RMSE_RGB_compare_Score + 0.19*cor_HSV_Result +\
                                     0.12 * Dominant_color_compare_score + 0.165*edge_all_Score
                     
                     
       
                    Score_number = 0.295 * Score_number_G + 0.245 *Part_Score + 0.175* Saliency_Score + 0.285*Score_position 
                 
                    compare_count = compare_count + 1
                    ############
                    # save Best similarity score:
                    ############
                    if Best_Score_number < Score_number:
                        Best_Score_number = Score_number

                        old_object_place = i_comp
                    else:
                        Best_Score_number = Best_Score_number
                    
    
                if Best_Score_number > 0.5:
           
                    Reid_features_id = old_object_place

                    Reid_features[:,:,:,Reid_features_id] = temp_Reid_features[:,:,:,cars_id_number]
                    
                    Reid_position[:,:,Reid_features_id] = temp_Reid_position[:,:,cars_id_number]
                                
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]


                    text_new = f"{labels[class_ids[i]]}-{Reid_features_id}: {Best_Score_number:.2f}"
                    (text_width, text_height) = cv2.getTextSize(text_new, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                    text_offset_x = x
                    text_offset_y = y - 5            
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                    overlay = image.copy()
                    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                    
                    cv2.putText(image, text_new, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

                    

                    list_now_frame.append(Reid_features_id)
                    
                    
                else:
                    print('Best_Score_number no include')
                    print(Best_Score_number)
                    print('old_object_place')
                    print(old_object_place)

                    # 1. add new Reid_features
                    ############## 
                    new_add_Reid_features = np.reshape(temp_Reid_features[:,:,:,cars_id_number] , (40,100,3,1))
                    Reid_features = np.append( Reid_features , new_add_Reid_features , axis = 3)
                    ##############
                    # 2. ReID_object
                    #############
                    new_add_REID_object = np.reshape(temp_image,(40,40,3,1))
                    ReID_object = np.append( ReID_object , new_add_REID_object , axis = 3)
                    ################
                    # add id number for new REID_object
                    #################
                    new_REID_number = ReID_object.shape[3] - 1
                    ######
                    ##############
                    # 3. Reid_position
                    ##############

                    Reid_position = np.append( Reid_position, temp_Reid_position, axis=2)

                    text_new = f"{labels[class_ids[i]]}-{new_REID_number}: {Best_Score_number:.2f}"
                    (text_width, text_height) = cv2.getTextSize(text_new, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                    text_offset_x = x
                    text_offset_y = y - 5            
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                    overlay = image.copy()
                    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

                    cv2.putText(image, text_new, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)                    
                

                    print('new_REID_number!!!!!!!!')
                    print(new_REID_number)
                    list_now_frame_new.append(new_REID_number)
                
                
                
                

                i_counter = i_counter +1

            cars_id_number = cars_id_number + 1
            
            

    if frame_counter > 0:

        print('list_now_frame')
        print(list_now_frame)
        print('array_compare[frame_counter-1]')
        print(array_compare[frame_counter-1])
        
        
        if type(array_compare[frame_counter-1]) is int:
            for compare_number in range(0,1):

                if array_compare[frame_counter-1] == list_now_frame[compare_number]:
                    counter_re_id = counter_re_id + 1
        else:

            if len(array_compare[frame_counter-1]) == 0:
                counter_re_id = counter_re_id
            else:
                #print('============================')
                for compare_number_count in range(0,len(array_compare[frame_counter-1])):

                    if array_compare[frame_counter-1][compare_number_count] == list_now_frame[compare_number_count]:

                        counter_re_id = counter_re_id + 1
                
        time_took_1 = time.perf_counter() - start  
        print('compare time! in one frame')
        print(time_took_1)

        time_all = time_all + time_took + time_took_1
                

    cv2.imwrite(crop_outpuut_save_path+str(frame_counter)+'.jpg',image)
    frame_counter = frame_counter + 1

    print('新的 frame的 compare number!!!!!!!!!!!!!!:')
    print(ReID_object.shape[3])
    compare_REID_number = ReID_object.shape[3]







for i_output in range(0,ReID_object.shape[3]):
    cv2.imwrite(id_save_path+str(i_output)+'_id'+'.jpg',ReID_object[:,:,:,i_output])
    

print('counter_re_id')
print(counter_re_id)

label_counter = 0 
for i in range(0,stop_frame-1):
    print('i number')
    print(i)
    if type(array_compare[i]) is int:
        label_counter = label_counter + 1
    else:
        
        label_counter = label_counter + len(array_compare[i])
    

print('label_counter')
print(label_counter)
print('acc of Re-ID:') 
print(counter_re_id/label_counter)
