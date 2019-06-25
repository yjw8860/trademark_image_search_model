import re, os, shutil
import numpy as np

folder = 'D:/trademark_image_search/data/img/symbol_and_character/'
query_folder = 'D:/trademark_image_search/data/img/query/'
ground_truth_folder = 'D:/trademark_image_search/data/img/ground_truth/'
img_list = os.listdir(folder)
reference = [name.split('_')[0] for name in img_list]
company_list = np.array(list(set([name.split('_')[0] for name in img_list])))
company_list = np.sort(company_list)
img_list = np.sort(np.array(img_list))
for company_name in company_list:
    print(company_name)
    index = [i for i, v in enumerate(reference) if company_name in v]
    basket = img_list[index]
    s_index = 0
    for img_name in basket:
        img_path = folder + img_name
        if s_index == 0:
            save_path = query_folder + img_name
            shutil.copy(img_path, save_path)
        else:
            if not os.path.exists(ground_truth_folder + re.sub('.jpg','', basket[0])):
                os.makedirs(ground_truth_folder + re.sub('.jpg','', basket[0]))
            save_path = ground_truth_folder + re.sub('.jpg','', basket[0]) + '/' + img_name
            shutil.copy(img_path, save_path)
        print(img_name)
        s_index += 1