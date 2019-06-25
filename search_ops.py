"""import basic library"""
import os, re
import numpy as np
import keras, cv2
import pandas as pd
from PIL import Image

"""import pre-trained model"""
from keras_applications.resnet import ResNet50

"""import keras layers"""
from keras import Sequential

"""import keras module"""
import keras.backend.tensorflow_backend as K

"""import yolo modules"""
from yolo_v2.frontend import YOLO
from yolo_ops import crop_img_via_yolo

def resnet_50():
    """Import pre-trained ResNet 50 model"""
    network = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                       backend=keras.backend, layers=keras.layers, models=keras.models,
                       utils=keras.utils)
    model = Sequential()
    model.add(network)
    model.summary()
    return model

def append_white_space(img):
    """
    Append white space to image

    input  : OpenCV type image
    return : rectangular image with appended white space
    """

    #convert Opencv image to PIL image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    #Compare width, height of image and define size that which is bigger
    width, height = img.size
    size = max(width, height)
    if not width == height:
        #Create background image color with black
        bg = Image.new('RGB', (size, size), (0,0,0))

        #Paste image to background image
        if width > height:
            height_start = int((width - height) / 2)
            bg.paste(img, (0, height_start))
        else:
            width_start = int((height - width) / 2)
            bg.paste(img, (width_start, 0))
    else:
        bg = img
    # convert PIL image image to Opencv image
    bg = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)

    return bg

def make_featuremap(model, img):
    """
    Make feature map with ResNet 50

    Input  : ResNet 50 model, image
    Return : Feature map
    """

    #Preprocessing image to make input data of ResNet 50
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3))

    #Make feature map
    feature = model.predict(img)
    feature = np.squeeze(feature, axis=0)

    return feature

def calculate_map(config):
    """
    Calculate mAP

    Input  : config
    Return : mAP
    """

    root_retrieval_result_folder = config['application']['retrieval_result_folder'] #Define retrieval result root directory
    root_ground_truth = config['application']['ground_truth_folder'] #Define ground truth root directory
    ground_truth_folder_list = os.listdir(root_ground_truth)

    mAP = [] #Define mAP list that return value of calculat_mAP function
    precision_list = [] #Define precision list
    for gt_folder in ground_truth_folder_list:
        retrieval_result_folder = root_retrieval_result_folder + gt_folder + '/'
        ground_truth_folder = root_ground_truth + gt_folder + '/'
        ground_truth_list = os.listdir(ground_truth_folder)
        retrieval_result_lists = os.listdir(retrieval_result_folder)

        #Image name of retrieval result is start with ordering number(ex: 0_image_name_1.jpg, 1_image_name_2.jpg).
        #If sort retrival result image list without append '0' properly, there are mistake for calculate mAP.
        #Therfore, in this code, '0' is padded to sort retrival result list properly.
        for index in range(len(retrieval_result_lists)):
            if len(retrieval_result_lists[index].split('_')[0])<5:
                for step in range(5):
                    retrieval_result_lists[index] = '0' + retrieval_result_lists[index]

        #Sorting retrieval result list
        retrieval_result_lists.sort()

        #To match with ground truth, ordering number is deleted
        retrieval_result_lists = ['_'.join(word.split('_')[1:]) for word in retrieval_result_lists]

        true_false = []
        target_list = []
        target = 0 #Denominator that used to calculate mAP
        for result in retrieval_result_lists:
            #If retrival result image name is in ground truth, return True, else return False
            check = result in ground_truth_list
            true_false.append(check)

            # If retrival result image name is in ground truth, plus 1 to denominator
            if check:
                target = target + 1
            target_list.append(target)

        num_list = np.array(list(range(len(true_false)))) + 1
        df = pd.DataFrame({'true_false':true_false, 'target':target_list, 'index':num_list})
        """
        df is looks like below
        
     true_false  target  index
         True       1      1
        False       1      2
        False       1      3
         True       2      4
        False       1      5
        False       1      6
        """
        df = df.loc[df['true_false']==True]
        """
        df is looks like below
        
     true_false  target  index
        True       1      1
        True       2      4
        """


        #There are some case that value of True is not exist in df.
        #Therefore, in this code, devide case that whether True value is exist in df or not.
        if True in true_false:
            precision = np.mean(np.array(df['target'].tolist()) / np.array(df['index'].tolist())) # ((1 / 1) + (2 / 4)) / 2
            precision_list.append(precision)
        #If there are no True value in df, precision value that 0 is appended to precision_list.
        else:
            precision_list.append(0)

    #Calculate mAP
    precision_list = np.mean(np.array(precision_list))
    mAP.append(precision_list)
    print(mAP)
    df = pd.DataFrame({'mAP':mAP})
    df.to_csv('./mAP.csv',index=False)

def _main_(config, save_feature=True, retrieval_images=True, calculate_MAP=True):
    """
    If save_feature is true, Make & save feature map
    If retrieval_images is true, start searching similar trademark images with query image
    If calculate_MAP is true, calcualting mAP

    Input  : config
    Return : feature map, retrieval result, mAP.csv
    """

    img_folder = config['application']['img_folder'] #Define trademark image folder
    feature_map_path = config['application']['feature_map_path'] #Define feature_map.npy path
    query_img_folder = config['application']['query_img_folder'] #Define query image folder directory
    weight_path = config['application']['trained_weights'] #Define well trained tiny yolo model weight path
    retrieval_result_folder = config['application']['retrieval_result_folder'] #Define retrieval result save folder

    #GPU-0 : import Tiny Yolo model
    with K.tf.device('/gpu:0'):
        yolo = YOLO(backend=config['model']['backend'],
                    input_size=config['model']['input_size'],
                    labels=config['model']['labels'],
                    max_box_per_image=config['model']['max_box_per_image'],
                    anchors=config['model']['anchors'])
        yolo.load_weights(weight_path)
    #GPU-1 : import ResNet50 model
    with K.tf.device('/gpu:1'):
        network = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                           backend=keras.backend, layers=keras.layers, models=keras.models,
                           utils=keras.utils)
        resnet_50 = Sequential()
        resnet_50.add(network)
        resnet_50.summary()

    ###Save Feature Map
    if save_feature:
        print('Start Save Feature map!')
        img_list = os.listdir(img_folder)
        feature_map = []
        for img_name in img_list:
            img_path = img_folder + img_name
            input_img = cv2.imread(img_path)
            input_img = 255 - input_img #reverse pixel value(ex: 255(white) --> 0, 0(black) --> 255
            boxes = yolo.predict(input_img)
            cropped_img = crop_img_via_yolo(input_img, boxes)
            cropped_img = append_white_space(cropped_img)
            feature_map.append(make_featuremap(resnet_50, cropped_img))
            print('Saving feature map of',img_name, 'is completed!')
        feature_map = np.array(feature_map)
        if not os.path.exists('./data/feature_map/'):
            os.makedirs('./data/feature_map/')
        np.save('./data/feature_map/feature_map.npy', feature_map)

    ###Retrieve Images
    if retrieval_images:
        if not os.path.exists(retrieval_result_folder):
            os.makedirs(retrieval_result_folder)
        print('Start retrieving trademark images')
        feature_map = np.load(feature_map_path)
        img_list = os.listdir(img_folder)
        query_img_list = os.listdir(query_img_folder)
        for query_img_name in query_img_list:
            query_img_path = query_img_folder + query_img_name
            query_img = cv2.imread(query_img_path)
            query_img = 255 - query_img #Reverse pixel value(ex: 255(white) --> 0, 0(black) --> 255
            boxes = yolo.predict(query_img) #OBJECT DETECTION WITH TINY YOLO MODEL
            cropped_img = crop_img_via_yolo(query_img, boxes) #Crop detected area
            cropped_img = append_white_space(cropped_img) #To prevent distortion of image, append white space to image
            feature = cv2.resize(cropped_img, (224, 224))
            feature = np.reshape(feature, (1, 224, 224, 3))
            y = resnet_50.predict(feature)
            distances = []
            feature_map_length = feature_map.shape[0]
            for step in range(feature_map_length):
                data = feature_map[step, :]
                distance = np.linalg.norm(data - y) #Calculate MSE(Mean Square Error)
                distances.append(distance)
            df = pd.DataFrame({'filename': img_list, 'distance': distances})
            df = df.sort_values(by=['distance']) #Sorting images with similarity(distance)
            sorted_filename = df['filename'].tolist()
            sorted_filename = sorted_filename[1:5] #Bring top-5 similar images

            #Save retrieval result
            for index in range(len(sorted_filename)):
                original_img_path = img_folder + sorted_filename[index]
                original_img = cv2.imread(original_img_path)
                if not os.path.exists(retrieval_result_folder + '/' + re.sub('.jpg','', query_img_name)):
                    os.makedirs(retrieval_result_folder + '/' + re.sub('.jpg','', query_img_name))
                save_path = retrieval_result_folder + '/' + re.sub('.jpg','', query_img_name) + '/' + str(index) + '_' + sorted_filename[index]
                cv2.imwrite(save_path, original_img)
            print('Retrieving', query_img_name, 'is completed!')
    ###Calculate
    if calculate_MAP:
        print('Start Calculate mAP')
        calculate_map(config)


