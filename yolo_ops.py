import cv2, re, os, json
import numpy as np
import xml.etree.cElementTree as ET
from PIL import Image
import pandas as pd


class crop_img():
    """
    To concatenate Symbol and character image, remove white space from image is performed.
    convert_sobel : Extract edge with Sobel method. (https://webnautes.tistory.com/1258)
    crop_horizental : Crop image horizentally
    crop_vertical : Crop image vertically
    crop_img :remove white space from image
    """
    def convert_sobel(self, img):
        try:
            img = cv2.UMat(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
            img_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

            img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
            img_sobel = cv2.fastNlMeansDenoising(img_sobel, None, 100, 7, 21)
            img_sobel = cv2.UMat.get(img_sobel)
        except cv2.error:
            img_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
            img_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

            img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
            img_sobel = cv2.fastNlMeansDenoising(img_sobel, None, 100, 7, 21)
            pass
        return img_sobel

    #Crop image horizentally
    def crop_horizental(self,original_img, refer_img):
        height = refer_img.shape[0]
        h_line_count = []

        for h_index in range(height):
            line = refer_img[h_index, :]
            sum = np.count_nonzero(line > 50)
            if sum > 0:
                sum = 1
            h_line_count.append(sum)

        h_line_count = np.array(h_line_count)
        index_0, = np.where(h_line_count == 0)
        length_1 = []
        for idx in range(len(index_0) - 1):
            basket = index_0[idx + 1] - index_0[idx]
            length_1.append(basket)
        length_1.append(1)
        length_1_percent = np.array(length_1) / height
        length_1_idx, = np.where(length_1_percent > 0.05)

        if len(index_0)==0:
            y_min = 0
            y_max = height-1
        else:
            if len(length_1_idx) == 0:
                y_min = 0
                y_max = height - 1
            elif len(length_1_idx) == 1:
                y_min = int(index_0[length_1_idx])
                y_max = int(index_0[length_1_idx] + max(length_1))
            elif len(length_1_idx) > 1:
                y_min = min(index_0[length_1_idx])
                length_address = length_1_idx[len(length_1_idx) - 1]
                length_basket = length_1[length_address]
                y_max = max(index_0[length_1_idx]) + length_basket

        crop_img = original_img[y_min:y_max, :, :]

        return crop_img

    def crop_vertical(self, original_img, refer_img):
        width = refer_img.shape[1]
        w_line_count = []

        for w_index in range(width):
            line = refer_img[:, w_index]
            sum = np.count_nonzero(line > 50)
            if sum > 0:
                sum = 1
            w_line_count.append(sum)

        w_line_count = np.array(w_line_count)
        index_0, = np.where(w_line_count == 0)
        length_1 = []
        for idx in range(len(index_0) - 1):
            basket = index_0[idx + 1] - index_0[idx]
            length_1.append(basket)
        length_1.append(1)
        length_1_percent = np.array(length_1) / width
        length_1_idx, = np.where(length_1_percent > 0.05)

        if len(index_0)==0:
            x_min = 0
            x_max = width-1
        else:
            if len(length_1_idx) == 0:
                x_min = 0
                x_max = width - 1
            elif len(length_1_idx) == 1:
                x_min = int(index_0[length_1_idx])
                x_max = int(index_0[length_1_idx] + max(length_1))
            elif len(length_1_idx) > 1:
                x_min = min(index_0[length_1_idx])
                length_address = length_1_idx[len(length_1_idx) - 1]
                length_basket = length_1[length_address]
                x_max = max(index_0[length_1_idx]) + length_basket

        crop_img = original_img[:, x_min:x_max, :]

        return crop_img

    def crop_img(self,original_img, refer_img):
        h_crop_img = self.crop_horizental(original_img, refer_img)
        h_refer_img = self.convert_sobel(h_crop_img)
        cropped_img = self.crop_vertical(h_crop_img, h_refer_img)

        return cropped_img


class concat_img():
    def compare_size(self, character_img, symbol_img):
        c_img = cv2.cvtColor(character_img, cv2.COLOR_BGR2RGB)
        c_img = Image.fromarray(c_img)
        c_x = c_img.size[0]
        c_y = c_img.size[1]

        s_img = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2RGB)
        s_img = Image.fromarray(s_img)

        s_x = s_img.size[0]
        s_y = s_img.size[1]

        final_x = max(c_x, s_x)
        final_y = max(c_y, s_y)

        compare_result = {'c_img':c_img,
                          's_img':s_img,
                          'c_width':c_x,
                          's_width':s_x,
                          'c_height':c_y,
                          's_height':s_y,
                          'x_max':final_x,
                          'y_max':final_y,
                          'width_sum':c_x + s_x,
                          'height_sum':c_y + s_y}

        return compare_result

    def concat_vertical(self,compare_result):
        width = compare_result['x_max']+100
        height = compare_result['height_sum']+100
        img = Image.new('RGB', (width, height), (255, 255, 255))
        index = np.array([0,1])
        np.random.shuffle(index)

        if index[0] == 0:
            c_x_min = int(np.random.uniform(low=0, high=width-compare_result['c_width']))
            s_x_min = int(np.random.uniform(low=0, high=width-compare_result['s_width']))
            c_y_min = int(np.random.uniform(low=0, high=height-(compare_result['c_height']+compare_result['s_height'])))
            s_y_min = int(np.random.uniform(low=c_y_min+compare_result['c_height'], high=height-compare_result['s_height']))
        else:
            c_x_min = int(np.random.uniform(low=0, high=width-compare_result['c_width']))
            s_x_min = int(np.random.uniform(low=0, high=width-compare_result['s_width']))
            s_y_min = int(np.random.uniform(low=0, high=height - (compare_result['c_height'] + compare_result['s_height'])))
            c_y_min = int(np.random.uniform(low=s_y_min+compare_result['s_height'], high=height-compare_result['c_height']))

        img.paste(compare_result['c_img'], (c_x_min, c_y_min))
        img.paste(compare_result['s_img'], (s_x_min, s_y_min))

        s_x_max = s_x_min + compare_result['s_width']
        s_y_max = s_y_min + compare_result['s_height']

        duplicate_result = {'width':width, 'height':height, 'x_min':s_x_min, 'y_min':s_y_min, 'x_max':s_x_max, 'y_max':s_y_max}

        return img, duplicate_result


    def concat_horizental(self,compare_result):
        width = compare_result['width_sum']+100
        height = compare_result['y_max']+100
        img = Image.new('RGB', (width, height), (255, 255, 255))
        index = np.array([0,1])
        np.random.shuffle(index)

        if index[0] == 0:
            c_x_min = int(np.random.uniform(low=0, high=width-(compare_result['c_width']+compare_result['s_width'])))
            s_x_min = int(np.random.uniform(low=c_x_min+compare_result['c_width'], high=width-compare_result['s_width']))
            c_y_min = int(np.random.uniform(low=0, high=height-compare_result['c_height']))
            s_y_min = int(np.random.uniform(low=0, high=height-compare_result['s_height']))
        else:
            s_x_min = int(np.random.uniform(low=0, high=width-(compare_result['c_width']+compare_result['s_width'])))
            c_x_min = int(np.random.uniform(low=s_x_min+compare_result['s_width'], high=width-compare_result['c_width']))
            c_y_min = int(np.random.uniform(low=0, high=height-compare_result['c_height']))
            s_y_min = int(np.random.uniform(low=0, high=height-compare_result['s_height']))

        img.paste(compare_result['c_img'], (c_x_min, c_y_min))
        img.paste(compare_result['s_img'], (s_x_min, s_y_min))

        s_x_max = s_x_min + compare_result['s_width']
        s_y_max = s_y_min + compare_result['s_height']

        duplicate_result = {'width':width, 'height':height, 'x_min':s_x_min, 'y_min':s_y_min, 'x_max':s_x_max, 'y_max':s_y_max}

        return img, duplicate_result

    def make_annotation(self,duplicate_result,img_name, img_path):
        root = ET.Element("annotation")
        ET.SubElement(root, 'filename').text = img_name
        ET.SubElement(root, 'path').text = img_path
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(duplicate_result['width'])
        ET.SubElement(size, 'height').text = str(duplicate_result['height'])
        object = ET.SubElement(root, 'object')
        ET.SubElement(object, 'name').text = 'symbol'
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(duplicate_result['x_min'])
        ET.SubElement(bndbox, 'ymin').text = str(duplicate_result['y_min'])
        ET.SubElement(bndbox, 'xmax').text = str(duplicate_result['x_max'])
        ET.SubElement(bndbox, 'ymax').text = str(duplicate_result['y_max'])
        tree = ET.ElementTree(root)
        xml_save_name = re.sub('.jpg', '.xml', img_path)
        tree.write(xml_save_name)


def make_train_test_data(make_data=True):
    crop = crop_img()
    concat = concat_img()
    if make_data:
        config_path = './config.json'
        with open(config_path) as config_buffer:
            config = json.loads(config_buffer.read())

        """DEFINE PATHS"""
        character_img_folder = config['data']['character_img_folder']  # 문자 이미지 원본
        character_crop_img_folder = config['data']['character_crop_img_folder']  # 여백이 제거된 문자 이미지
        symbol_img_folder = config['data']['symbol_img_folder']  # 도형 이미지 원본
        concat_img_folder = config['data']['concat_img_folder'] # 문자 이미지와 도형 이미지가 합쳐져서 저장될 경로
        annotation_folder = config['data']['annotation_folder']  # annotation이 저장될 경로

        """REMOVE WHITE SPACE FROM CHARACTER IMAGE"""
        character_img_list = np.array(os.listdir(character_img_folder))
        for img_name in character_img_list:
            img_path = character_img_folder + img_name
            original_img = cv2.imread(img_path)
            refer_img = crop.convert_sobel(img = original_img)
            f_img = crop.crop_img(original_img, refer_img)
            save_path = character_crop_img_folder + img_name
            cv2.imwrite(save_path, f_img)

        """MAKE CONCAT IMAGE"""
        symbol_img_list = np.array(os.listdir(symbol_img_folder))
        np.random.seed(2018121604)
        np.random.shuffle(symbol_img_list) #shuffle symbol image file name list
        devied_point = int(len(symbol_img_list) * 0.7)
        train_list = symbol_img_list[:devied_point] #define train data
        test_list = symbol_img_list[devied_point:] #define test data

        #make train data
        v_h_index = np.array([0,1])
        for img_name in train_list:
            save_folder = concat_img_folder + 'train/'
            annotation_save_folder = annotation_folder + 'train/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if not os.path.exists(annotation_save_folder):
                os.makedirs(annotation_save_folder)

            symbol_img_path = symbol_img_folder + img_name
            symbol_original_img = cv2.imread(symbol_img_path)
            symbol_refer_img = crop.convert_sobel(symbol_original_img)
            symbol_cropped_img = crop.crop_img(symbol_original_img, symbol_refer_img)

            np.random.shuffle(character_img_list)
            character_img_path = character_crop_img_folder + character_img_list[0]
            character_img = cv2.imread(character_img_path)
            compare_result = concat.compare_size(character_img, symbol_cropped_img)
            np.random.shuffle(v_h_index)

            character_filename_re = re.sub('.jpg', '', character_img_list[0])
            symbol_filename_re = re.sub('.jpg', '', img_name)
            c_save_name = symbol_filename_re + '_' + str(character_filename_re) + '.jpg'
            if v_h_index[0] == 0:
                v_img, duplicate_result = concat.concat_vertical(compare_result)
                img_save_path = save_folder + c_save_name
                v_img.save(img_save_path)

            else:
                h_img, duplicate_result = concat.concat_horizental(compare_result)
                img_save_path = save_folder + c_save_name
                h_img.save(img_save_path)

            annotation_save_path = annotation_save_folder + c_save_name
            concat.make_annotation(duplicate_result, c_save_name, annotation_save_path)

        #make test data
        v_h_index = np.array([0,1])
        for img_name in test_list:
            save_folder = concat_img_folder + 'test/'
            annotation_save_folder = annotation_folder + 'test/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if not os.path.exists(annotation_save_folder):
                os.makedirs(annotation_save_folder)

            symbol_img_path = symbol_img_folder + img_name
            symbol_original_img = cv2.imread(symbol_img_path)
            symbol_refer_img = crop.convert_sobel(symbol_original_img)
            symbol_cropped_img = crop.crop_img(symbol_original_img, symbol_refer_img)

            np.random.shuffle(character_img_list)
            character_img_path = character_crop_img_folder + character_img_list[0]
            character_img = cv2.imread(character_img_path)
            compare_result = concat.compare_size(character_img, symbol_cropped_img)
            np.random.shuffle(v_h_index)

            character_filename_re = re.sub('.jpg', '', character_img_list[0])
            symbol_filename_re = re.sub('.jpg', '', img_name)
            c_save_name = symbol_filename_re + '_' + str(character_filename_re) + '.jpg'
            if v_h_index[0] == 0:
                v_img, duplicate_result = concat.concat_vertical(compare_result)
                img_save_path = save_folder + c_save_name
                v_img.save(img_save_path)

            else:
                h_img, duplicate_result = concat.concat_horizental(compare_result)
                img_save_path = save_folder + c_save_name
                h_img.save(img_save_path)

            annotation_save_path = annotation_save_folder + c_save_name
            concat.make_annotation(duplicate_result, c_save_name, annotation_save_path)

# class apply_yolo():
def negative_to_zero(value):
    if value < 0:
        value = 0
    return value

def crop_img_via_yolo(input_img, boxes):
    image_h, image_w, _ = input_img.shape
    if len(boxes) == 0:
        crop_img = input_img
    elif len(boxes) == 1:
        for box in boxes:
            f_xmin = negative_to_zero(int(box.xmin * image_w))
            f_ymin = negative_to_zero(int(box.ymin * image_h))
            f_xmax = negative_to_zero(int(box.xmax * image_w))
            f_ymax = negative_to_zero(int(box.ymax * image_h))
            crop_img = input_img[f_ymin:f_ymax, f_xmin:f_xmax]
    else:
        xmin_list = []
        xmax_list = []
        ymin_list = []
        ymax_list = []
        score_list = []
        for box in boxes:
            xmin = int(box.xmin * image_w)
            ymin = int(box.ymin * image_h)
            xmax = int(box.xmax * image_w)
            ymax = int(box.ymax * image_h)
            xmin_list.append(xmin)
            xmax_list.append(xmax)
            ymin_list.append(ymin)
            ymax_list.append(ymax)
            score_list.append(box.get_score())
        df = pd.DataFrame(
            {'xmin': xmin_list, 'xmax': xmax_list, 'ymin': ymin_list, 'ymax': ymax_list, 'score': score_list})
        df = df[['score','xmax', 'xmin', 'ymax', 'ymin']]
        df = df.sort_values('score', ascending=False)

        f_xmax = negative_to_zero(df.iloc[0, 1])
        f_xmin = negative_to_zero(df.iloc[0, 2])
        f_ymax = negative_to_zero(df.iloc[0, 3])
        f_ymin = negative_to_zero(df.iloc[0, 4])
        crop_img = input_img[f_ymin:f_ymax, f_xmin:f_xmax]

    return crop_img