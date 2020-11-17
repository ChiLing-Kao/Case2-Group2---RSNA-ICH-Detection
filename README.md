# Case2-Group2---RSNA-ICH-Detection
### I. Introduction
###### **A. Background**
Intracranial hemorrhage (ICH) is a serious health problem often requiring rapid and intensive treatment. Identifying any hemorrhage present is a critical step in treating the patient. Radiologists must rapidly review images of the patient’s cranium to look for the presence, location and type of hemorrhage. 
Diagnosis requires an urgent procedure. When a patient shows acute neurological symptoms such as severe headache or loss of consciousness, highly trained specialists review medical images of the patient’s cranium to look for the presence, location and type of hemorrhage. The process is complicated and often time consuming.

###### **B. Problems**
When structuring a deep learning model (classification, segmentation, object detection), there are lots of problems with medical images, such as, noise, non-standardized acquisition, etc. We have to do some preprocessing and put them into the deep learning model. 

###### **C. Goals**
* Build a model to detect acute intracranial hemorrhage and its subtypes.
* Compare different image preprocessing methods and put them into the model.

### II. Data Introduction
The data provided by the Radiological Society of North America in collaboration with members of the American Society of Neuroradiology and MD.ai.. There are 6,000 training data and 600 test data which split into 6 categories – healthy, intraparenchymal, intraventricular, subarachnoid, subdural and epidural. 
![image](https://github.com/ChiLing-Kao/Case2-Group2---RSNA-ICH-Detection/blob/main/image.png)

### III. Requirements
Python 3.6.8 or later with all requirements.txt dependencies installed, including. To install run:
```js
$ pip install -r requirements.txt
```

### IV. Dataset
The data is in the **raw_data** folder and **data_preprocessing** folder. F
###### **A. raw_data** : There are **TrainingData** and **TestingData**. In **TrainingData**, the images split into 6 categories folders with 15 images respectively. In **TestingData**, there are 15 images. The images are all in DICOM form.
###### **B. data_preprocessing** : There are 7 folders which are used in 7 different image preprocessing. The images are all in PNG form and are 20 images respectively.
* 0_Dicom2PNG
* 1_ROI
* 2_ROI+CLAHE
* 3_ROI+CLAHE+Bilateral
* 4_ROI+noblank
* 5_ROI+noblank+CLAHE
* 6_ROI+noblank+CLAHE+Bilateral

### IV. Step1. Image Preprocessing - Step1_Image_Preprocessing.py**
* The file contain 5 image preprocessing method - Dicom to png, ROI, CLAHE, Without white Margin, Bilateral Filtering.

### V. Step2. EfficientNet Model - Step2_EfficientNet_Model.ipynb**
* Use EfficientNet B7 to train the model and output the result.
