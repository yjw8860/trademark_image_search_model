import os, re
from PIL import Image
import numpy as np

character_path = 'D:/data/tm_190422/img/character_unique_crop/'
symbol_path = 'D:/data/tm_190422/img/symbol_unique_crop/'
input_save_folder = 'D:/data/tm_190422/img/pix2pix_concat_symbol_character/input/'
target_save_folder = 'D:/data/tm_190422/img/pix2pix_concat_symbol_character/target/'

if not os.path.exists(input_save_folder):
    os.makedirs(input_save_folder)
if not os.path.exists(target_save_folder):
    os.makedirs(target_save_folder)

character_filename_list= os.listdir(character_path)
symbol_filename_list = os.listdir(symbol_path)

def compare_size(character_img_path, symbol_img_path):
    c_img = Image.open(character_img_path)
    c_x = c_img.size[0]
    c_y = c_img.size[1]

    s_img = Image.open(symbol_img_path)
    s_x = s_img.size[0]
    s_y = s_img.size[1]

    final_x = max(c_x, s_x)
    final_y = max(c_y, s_y)

    compare_result = {'c_img':c_img, 's_img':s_img,'c_width':c_x, 's_width':s_x, 'c_height':c_y, 's_height':s_y, 'x_max':final_x, 'y_max':final_y, 'width_sum':c_x + s_x, 'height_sum':c_y + s_y}

    return compare_result

def concat_vertical(compare_result):
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


def concat_horizental(compare_result):
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


def concat_vertical_2(compare_result, input_save_folder, target_save_folder, s_img_name):
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

    img.paste(compare_result['s_img'], (s_x_min, s_y_min))
    save_path = target_save_folder + s_img_name
    img.save(save_path)
    img.paste(compare_result['c_img'], (c_x_min, c_y_min))
    save_path = input_save_folder + s_img_name
    img.save(save_path)



def concat_horizental_2(compare_result, input_save_folder, target_save_folder, s_img_name):
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

    img.paste(compare_result['s_img'], (s_x_min, s_y_min))
    save_path = target_save_folder + s_img_name
    img.save(save_path)
    img.paste(compare_result['c_img'], (c_x_min, c_y_min))
    save_path = input_save_folder + s_img_name
    img.save(save_path)


# def concat_img_190402():
#     for character_filename in character_filename_list:
#         character_img_path = os.path.join(character_path, character_filename)
#         for symbol_filename in symbol_filename_list:
#             print(symbol_filename)
#             symbol_img_path = os.path.join(symbol_path, symbol_filename)
#             compare_result = compare_size(character_img_path, symbol_img_path)
#             v_img, v_duplicate_result = concat_vertical(compare_result)
#             h_img, h_duplicate_result = concat_horizental(compare_result)
#
#             character_filename_re = re.sub('.jpg', '', character_filename)
#             symbol_filename_re = re.sub('.jpg','',symbol_filename)
#             v_save_name = str(character_filename_re) + '_' + symbol_filename_re + '_v.jpg'
#             v_save_path = save_folder + v_save_name
#             h_save_name = str(character_filename_re) + '_' + symbol_filename_re + '_h.jpg'
#             h_save_path = save_folder + h_save_name
#             v_img.save(v_save_path)
#             h_img.save(h_save_path)


v_h_index = np.array([0,1])
name_index = 0
for symbol_filename in symbol_filename_list:
    try:
        symbol_img_path = os.path.join(symbol_path, symbol_filename)
        character_filename_list = np.array(character_filename_list)
        np.random.shuffle(character_filename_list)
        character_img_path = os.path.join(character_path, character_filename_list[0])
        symbol_img_path = os.path.join(symbol_path, symbol_filename)
        compare_result = compare_size(character_img_path, symbol_img_path)
        np.random.shuffle(v_h_index)

        character_filename_re = re.sub('.jpg', '', character_filename_list[0])
        symbol_filename_re = re.sub('.jpg', '', symbol_filename)

        if v_h_index[0] == 0:
            concat_vertical_2(compare_result, input_save_folder, target_save_folder, symbol_filename)
        else:
            concat_horizental_2(compare_result,input_save_folder, target_save_folder, symbol_filename)
        print(symbol_filename)
    except OSError as e:
      pass


