# Multi-viewpoint-Patterns-and-Occlusions-Handling-Using-Hybrid-Features-for-Vehicle-Tracking

## Source code of vehicle tracking and Re-ID

### Authors



### Get Started
1. cd the folder where you want to download this repo
2. Run git clone https://github.com/dddd247/Multi-viewpoint-Patterns-and-Occlusions-Handling-Using-Hybrid-Features-for-Vehicle-Tracking.git
3. Install dependencise:
   * python >= 3.6
   * pytorch >= 0.4
   * opencv-contrib-python >= 3.4.1.15
   * opencv-python >= 3.4.1.15
   * colorsys
   * time
   * scipy
   * math
   * numpy
   * os, sys
4. Prepare dataset, VehicleMVO,
   Download the dataset from https://drive.google.com/drive/folders/1OER0lGggqaUzWJdxCDgxYkGsofvf8voU?usp=sharing
   you can put them into the file "Souce_video_bdd100k_video"
   #### There are ten different videos in this drive, we separate them into 2 parts: Day and Night coniditions.
   * Day_normal_1 -> Day_normal_5.mp4
   * night_video_1 -> night_video_5.mp4
5. Create a file and name it weight.
   Download the object detector weight, "yolov3.weight", and then put it into the file 
   
   
#### Run
(1) Run the code on the command line. 
    
    python Day_tracking.py
    
(2) You can change the similarity threshold in the program at line 1152, we set it 0.5 

#### useful tips
(1) The details of global feature, partial similarity, salient section, position correlation are in compare_similarity_use.py

(2) The original program uses the video frames from Day_normal_1.mp4 to conduct the vehicle tracking and re-identification. 
    If you want to test other videos from the datasets, VehicleMVO, You must follow the steps as follows:
    
    (1) you need to convert them into video frames first.
    
    (2) Go into the file place, ./final_real_world_detect_output/Test/Day_normal, and create the file structure like this.
        * Day_normal_detect_output
        * Day_normal_each_id_sace
        * Day_normal_original_video_output
        * Day_normal_read_world_first_two_frame_compair_pair
        
     (3) Put your pre-process video frames into ./Day_normal_original_video_output
     
     (4) change the code in Day_tracking.py at line 39, 147, 153, 178
     
     (5) change the labels at line 50 and set the number of stop frame at line 184.

    
   
