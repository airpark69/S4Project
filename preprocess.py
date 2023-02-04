import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import yaml


# xml에서 파싱하여 image_list와 label을 읽어오고 기존의 label에 부여된 숫자와 비교한 뒤, 추가될 부분은 추가합니다.
def load_xml(xml_path ,label_path="./label.json"):

  xml_anot = ET.parse(xml_path)
  root = xml_anot.getroot()

  meta = root.find('meta')

  ## image 파일들의 정보가 담긴 list
  image_list = root.findall('image')

  task = meta.find('task')

  labels = task.find('labels')

  label_list = []

  for label in labels:
    label_list.append(label[0].text)

  try:
    with open(label_path, "r") as json_file:
        label_dict = json.load(json_file)
  except:
    print("label.json 새로 생성됨")
    label_dict = {}

  label_split_list = ['person','scooter','bicycle','motorcycle']
    
  for label in label_list:
    if label in label_split_list:
      if label_dict.get(label) is None:
        label_dict[label] = len(label_dict)

  ## label_dict를 저장, 이후에 로드해서 업데이트하면서 사용
  with open(label_path,'w') as f:
    json.dump(label_dict, f, indent=4)

  return image_list, label_dict


## unit_square 기준에서 normalize -> YOLO를 위한 방식은 아님
def normalize_polygon(polygon):
    polygon = np.array(polygon)

    # Translate the polygon so that its centroid is at the origin
    centroid = np.mean(polygon, axis=0)
    polygon -= centroid

    # Scale the polygon so that its bounding box fits in a unit square
    min_arr = np.array(np.min(polygon, axis=0))
    max_arr = np.array(np.max(polygon, axis=0))
    polygon = (polygon - min_arr) / (max_arr - min_arr)

    return polygon

## Input_size 기준 -> YOLO를 위한 방식
def normalize_polygon_yolov5(polygon, image_size):
    polygon = np.array(polygon)

    # Scale the polygon so that its bounding box fits in the input image
    polygon = polygon / np.array(image_size)

    return polygon


def yolo_label_txt_transform(polygons, label_dict):
  img_txt = ''

  for polygon in polygons:
      img_label = polygon.attrib['label']
      if label_dict.get(img_label) is None:
        continue
      img_polygon = polygon.attrib['points']

      point_list = img_polygon.split(';')
      polygon = [x.split(',') for x in point_list]
      polygon = [[float(point) for point in x] for x in polygon]
      
      # Normalize the polygon / input_size = 1920 1080 고정
      normalized_polygon = normalize_polygon_yolov5(polygon, [1920, 1080])

      ## label_dict와 비교해서 label을 숫자로 입력
      img_txt += str(label_dict[img_label]) + " "

      for poly in normalized_polygon:
        for point in poly:
          img_txt += str(point) + " "

      img_txt = img_txt.rstrip()
      img_txt += "\n"
  
  return img_txt[:-1]


def make_label_txt(folder, img_list, label_dict):
  # img_list 내의 모든 image에 대해서 label 값에 해당하는 폴리곤 마스킹 txt를 만든다


  # labels 폴더가 없을 경우 생성
  labels_path = folder + "/labels"
  if not os.path.exists(labels_path):
    os.makedirs(labels_path)
    print(f"Folder {labels_path} created.")
  else:
    print(f"Folder {labels_path} already exists.")

  for img in img_list:
    img_name = img.attrib['name']
    img_path = folder+"/"+img_name
    label_path = labels_path + "/" + img_name + ".txt"
    polygons = img.findall('polygon')
    label_txt = yolo_label_txt_transform(polygons, label_dict)

    if label_txt == '':
      ## 이미지에 필요한 라벨이 없을 경우 삭제
      try:
        os.remove(img_path)
        print(f"{img_path} has been deleted.")
      except FileNotFoundError:
        print(f"{img_path} does not exist.")
      continue

    with open(label_path, "w") as file:
        file.write(label_txt)


def preprocessing(folder):

    def xml_parsing_and_make_txt(folder):
    # Get a list of all XML files in the folder
        xml_files = [file for file in os.listdir(folder) if file.endswith('.xml')]

        for xml_file in xml_files:
            file_path = os.path.join(folder, xml_file)


        img_list, label_dict = load_xml(file_path)

        make_label_txt(folder, img_list, label_dict)
  
  # 해당 폴더 내의 모든 폴더를 탐색하고 필요없는 라벨만 있는 이미지 삭제 및 라벨 변환

    for dir in os.listdir(folder):
        sub_dir_path = os.path.join(folder, dir)
        print(sub_dir_path)
        if os.path.isdir(sub_dir_path):
    
            xml_parsing_and_make_txt(sub_dir_path)
            print(f"{sub_dir_path} 폴더 변환 완료!")
    


def make_data_yaml():
  # xml에서 파싱한 label을 기준으로 data.yaml을 설정
  # train_path와 val_path는 사용 환경에 따라 다르게 지정해줘야 함

  try:
    with open('./label.json', "r") as json_file:
        label_dict = json.load(json_file)
  except:
    print("에러 : label.json 존재하지 않음")
    return

  data = {}

  train_path = "../train/images"
  val_path = "../valid/images"

  data['names'] = list(label_dict.keys())
  data['nc'] = len(label_dict)
  data['train'] = train_path
  data['val'] = val_path

  with open('./data.yaml', 'w') as f:
    yaml.dump(data, f)
  print("data.yaml 생성 완료!")  