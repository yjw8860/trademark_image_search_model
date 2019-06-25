import os
import xml.etree.ElementTree as ET
import pandas as pd

train_xml_folder = 'D:/data/tm_190422/annotations/test/'
train_xml_name_list = os.listdir(train_xml_folder)

filename_list = []
width_list = []
height_list = []
class_list = []
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []


for xml_name in train_xml_name_list:
    xml_path = train_xml_folder + xml_name
    xml = ET.parse(xml_path)
    root = xml.getroot()
    # for filename in root.findall('filename'):
    #     filename_list.append(filename.text)
    for child in root:
        if child.tag == 'filename':
            filename_list.append(child.text)
        elif child.tag == 'size':
            for child_1 in child:
                if child_1.tag == 'width':
                    width_list.append(child_1.text)
                else:
                    height_list.append(child_1.text)
        elif child.tag == 'object':
            for child_1 in child:
                if child_1.tag == 'name':
                    class_list.append(child_1.text)
                else:
                    for child_2 in child_1:
                        if child_2.tag == 'xmin':
                            xmin_list.append(child_2.text)
                        elif child_2.tag == 'ymin':
                            ymin_list.append(child_2.text)
                        elif child_2.tag == 'xmax':
                            xmax_list.append(child_2.text)
                        else:
                            ymax_list.append(child_2.text)
    print(xml_name)

df = pd.DataFrame({'filename':filename_list, 'width':width_list, 'height':height_list, 'class':class_list, 'xmin':xmin_list, 'ymin':ymin_list, 'xmax':xmax_list, 'ymax':ymax_list})
cols = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
df = df[cols]
df.to_csv('./test_label.csv', index=False)