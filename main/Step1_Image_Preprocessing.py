import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt                  

#%%%
# 1. Laod Dicom image 
# (1) Training data path
# https://www.kaggle.com/srsteinkamp/intuitions-ideas-basic-preprocessing
# https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/2DNet/src/train.py

directory = 'D:/Chi/Homework/case2/data'

healthy_dir = directory + '/TrainingData/healthy/'
intraparenchymal_dir = directory + '/TrainingData/intraparenchymal/' #實質內
intraventricular_dir = directory + '/TrainingData/intraventricular/' #腦室內
subarachnoid_dir = directory + '/TrainingData/subarachnoid/'         #蛛網膜下腔
subdural_dir = directory + '/TrainingData/subdural/'                 #硬膜下
epidural_dir = directory + '/TrainingData/epidural/'                 #硬膜外
test_dir = directory + '/TestingData/'                 

# (2) Load dicom image function
import pydicom 

def get_first_dicom_field(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)
    
def LoadDicomImage(path):
    # 1-1. Dicom Information
    img_dicom = pydicom.read_file(path) 
    # 1-2. Dicom Image
    img = img_dicom.pixel_array 
    
    # 2. Get dicom information to process the image (直接用.pixel_array樣子會跑掉)    
    window_center = get_first_dicom_field(img_dicom.WindowCenter)
    window_width = get_first_dicom_field(img_dicom.WindowWidth)
    intercept = img_dicom.RescaleIntercept
    slope = img_dicom.RescaleSlope

    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    
    # 3. Normalize the image
    minimum, maximum = img.min(), img.max()
    img = ((img - minimum) / (maximum - minimum))*255
    
    return img

# (3) Save imagein png form
save_dir = 'D:/Chi/Homework/case2/data_preprocessing/'

# create 6 categories file
os.makedirs(save_dir + '/step1_Dicom2PNG/healthy/')
os.makedirs(save_dir + '/step1_Dicom2PNG/intraparenchymal/') #實質內
os.makedirs(save_dir + '/step1_Dicom2PNG/intraventricular/') #腦室內
os.makedirs(save_dir + '/step1_Dicom2PNG/subarachnoid/')     #蛛網膜下腔
os.makedirs(save_dir + '/step1_Dicom2PNG/subdural/')         #硬膜下
os.makedirs(save_dir + '/step1_Dicom2PNG/epidural/')         #硬膜外
os.makedirs(save_dir + '/step1_Dicom2PNG_test/')         

# Preprocess the image and save
def Preprocessing(category_path, save_folder_path):
    file_name = os.listdir(category_path)
    for i in range(len(file_name)):
        path = category_path + file_name[i]
        img = LoadDicomImage(path)
        cv2.imwrite(save_folder_path + str(file_name[i].split('.')[0]) + '.png', img)

# A. healthy
category_path = healthy_dir
save_folder_path = save_dir + '/step1_Dicom2PNG/healthy/'
Preprocessing(category_path, save_folder_path)

# B. intraparenchymal
category_path = intraparenchymal_dir
save_folder_path = save_dir + '/step1_Dicom2PNG/intraparenchymal/'
Preprocessing(category_path, save_folder_path)

# C. intraventricular
category_path = intraventricular_dir
save_folder_path = save_dir + '/step1_Dicom2PNG/intraventricular/'
Preprocessing(category_path, save_folder_path)

# D. subarachnoid
category_path = subarachnoid_dir
save_folder_path = save_dir + '/step1_Dicom2PNG/subarachnoid/'
Preprocessing(category_path, save_folder_path)

# E. subdural
category_path = subdural_dir
save_folder_path = save_dir + '/step1_Dicom2PNG/subdural/'
Preprocessing(category_path, save_folder_path)

# F. epidural
category_path = epidural_dir
save_folder_path = save_dir + '/step1_Dicom2PNG/epidural/'
Preprocessing(category_path, save_folder_path)

# Test
category_path = test_dir
save_folder_path = save_dir + '/step1_Dicom2PNG_test/'
Preprocessing(category_path, save_folder_path)

   
#%% Get the ROI image
def ROI(category_path, save_folder_path):
    
    file_name = os.listdir(category_path)
    
    for i in range(len(file_name)):
        img = cv2.imread(category_path+file_name[i]) 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        contours,hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        area_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            area_list.append(area)
        
        idxs = area_list.index(max(area_list))
        ROI_mask = cv2.drawContours(np.zeros(img.shape), [contours[idxs]], 0, (255, 255, 255), -1) 
        ROI_mask[ROI_mask == 255] = 1
        img[np.where(ROI_mask == 0)] = 0                                     
        
        cv2.imwrite(save_folder_path + str(file_name[i].split('.')[0]) + '.png', img)
    
# (2) Save imagein png form
save_dir = 'D:/Chi/Homework/case2/data_preprocessing/'

# create 6 categories file
os.makedirs(save_dir + '/step2_ROI/healthy/')
os.makedirs(save_dir + '/step2_ROI/intraparenchymal/') #實質內
os.makedirs(save_dir + '/step2_ROI/intraventricular/') #腦室內
os.makedirs(save_dir + '/step2_ROI/subarachnoid/')     #蛛網膜下腔
os.makedirs(save_dir + '/step2_ROI/subdural/')         #硬膜下
os.makedirs(save_dir + '/step2_ROI/epidural/')         #硬膜外
os.makedirs(save_dir + '/step2_ROI_test/')         

# A. healthy
category_path_healthy = save_dir + '/step1_Dicom2PNG/healthy/'
save_folder_path_healthy = save_dir + '/step2_ROI/healthy/'
ROI(category_path_healthy, save_folder_path_healthy)

# B. intraparenchymal
category_path_intraparenchymal = save_dir + '/step1_Dicom2PNG/intraparenchymal/'
save_folder_path_intraparenchymal = save_dir + '/step2_ROI/intraparenchymal/'
ROI(category_path_intraparenchymal, save_folder_path_intraparenchymal)

# C. intraventricular
category_path_intraventricular = save_dir + '/step1_Dicom2PNG/intraventricular/'
save_folder_path_intraventricular = save_dir + '/step2_ROI/intraventricular/'
ROI(category_path_intraventricular, save_folder_path_intraventricular)

# D. subarachnoid
category_path_subarachnoid = save_dir + '/step1_Dicom2PNG/subarachnoid/'
save_folder_path_subarachnoid = save_dir + '/step2_ROI/subarachnoid/'
ROI(category_path_subarachnoid, save_folder_path_subarachnoid)

# E. subdural
category_path_subdural = save_dir + '/step1_Dicom2PNG/subdural/'
save_folder_path_subdural = save_dir + '/step2_ROI/subdural/'
ROI(category_path_subdural, save_folder_path_subdural)

# F. epidural
category_path_epidural = save_dir + '/step1_Dicom2PNG/epidural/'
save_folder_path_epidural = save_dir + '/step2_ROI/epidural/'
ROI(category_path_epidural, save_folder_path_epidural)

# test
category_path_test = save_dir + '/step1_Dicom2PNG_test/'
save_folder_path_test = save_dir + '/step2_ROI_test/'
ROI(category_path_test, save_folder_path_test)
 
#%% 2. Imaging Preprocessing
# https://medium.com/@cindylin_1410/%E6%B7%BA%E8%AB%87-opencv-%E7%9B%B4%E6%96%B9%E5%9C%96%E5%9D%87%E8%A1%A1%E5%8C%96-ahe%E5%9D%87%E8%A1%A1-clahe%E5%9D%87%E8%A1%A1-ebc9c14a8f96
# https://www.geeksforgeeks.org/python-bilateral-filtering/
# 李立宗 - 科班出身的AI人必修課：OpenCV影像處理 使用python
# Rafael C. Gonzalez , Richard E. Woods - Digital Image Processing, 4/e (GE-Paperback)


def Threshold_CLAHE_Bilateral(category_path,save_folder_path):
    
    file_name = os.listdir(category_path)
    for i in range(len(file_name)):
        
        img = cv2.imread(category_path+file_name[i])  
        
        # (1) Threshold - 去掉白邊
        t,img_thres = cv2.threshold(img, 254, 255, cv2.THRESH_TOZERO_INV)
       
        # (2) CLAHE - 自適應的局部的直方圖均衡化createCLAHE
        img_thres_gray = cv2.cvtColor(img_thres,cv2.COLOR_BGR2GRAY)
        CLAHE = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        img_clahe = CLAHE.apply(img_thres_gray)
        
        # (3) Bilateral Filter
        img_bilateral = cv2.bilateralFilter(img_clahe, 20, 50, 50)
                
        cv2.imwrite(save_folder_path + str(file_name[i].split('.')[0]) + '.png', img_bilateral)


# (2) Save imagein png form
save_dir = 'D:/Chi/Homework/case2/data_preprocessing/'

# create 6 categories file
os.makedirs(save_dir + '/step3_CV2_processing/healthy/')
os.makedirs(save_dir + '/step3_CV2_processing/intraparenchymal/') #實質內
os.makedirs(save_dir + '/step3_CV2_processing/intraventricular/') #腦室內
os.makedirs(save_dir + '/step3_CV2_processing/subarachnoid/')     #蛛網膜下腔
os.makedirs(save_dir + '/step3_CV2_processing/subdural/')         #硬膜下
os.makedirs(save_dir + '/step3_CV2_processing/epidural/')         #硬膜外
os.makedirs(save_dir + '/step3_CV2_processing_test')         #硬膜外

# A. healthy
category_path_healthy = save_dir + '/step2_ROI/healthy/'
save_folder_path_healthy = save_dir + '/step3_CV2_processing/healthy/'
Threshold_CLAHE_Bilateral(category_path_healthy,save_folder_path_healthy)
                          
# B. intraparenchymal
category_path_intraparenchymal = save_dir + '/step2_ROI/intraparenchymal/'
save_folder_path_intraparenchymal = save_dir + '/step3_CV2_processing/intraparenchymal/'
Threshold_CLAHE_Bilateral(category_path_intraparenchymal,save_folder_path_intraparenchymal)
                          
# C. intraventricular
category_path_intraventricular = save_dir + '/step2_ROI/intraventricular/'
save_folder_path_intraventricular = save_dir + '/step3_CV2_processing/intraventricular/'
Threshold_CLAHE_Bilateral(category_path_intraventricular,save_folder_path_intraventricular)

# D. subarachnoid
category_path_subarachnoid = save_dir + '/step2_ROI/subarachnoid/'
save_folder_path_subarachnoid = save_dir + '/step3_CV2_processing/subarachnoid/'
Threshold_CLAHE_Bilateral(category_path_subarachnoid,save_folder_path_subarachnoid)

# E. subdural
category_path_subdural = save_dir + '/step2_ROI/subdural/'
save_folder_path_subdural = save_dir + '/step3_CV2_processing/subdural/'
Threshold_CLAHE_Bilateral(category_path_subdural,save_folder_path_subdural)

# F. epidural
category_path_epidural = save_dir + '/step2_ROI/epidural/'
save_folder_path_epidural = save_dir + '/step3_CV2_processing/epidural/'
Threshold_CLAHE_Bilateral(category_path_epidural,save_folder_path_epidural)

# Test
category_path_test = save_dir + '/step2_ROI_test/'
save_folder_path_test = save_dir + '/step3_CV2_processing_test/'
Threshold_CLAHE_Bilateral(category_path_test,save_folder_path_test)