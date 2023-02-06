import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter


# label 불균형 확인하기 위한 counter
label_cv = Counter()

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

  label_split_list = ['person','scooter','bicycle','motorcycle', 'sidewalk', 'roadway', 'alley', 'bike_lane']
    
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
      # 해당 라벨이 없을 경우 이 폴리곤은 제외한다
      if label_dict.get(img_label) is None:
        continue
      label_cv.update([img_label])
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


  # 폴더가 없으면 만드는 함수
  def make_dir(dir_name):
    dir_path = os.path.join(os.getcwd(),dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Folder {dir_path} created.")
    else:
        print(f"Folder {dir_path} already exists.")
    
    return dir_path
  
  # dataset 폴더가 없을 경우 생성
  make_dir("dataset")
  # labels 폴더가 없을 경우 생성
  labels_path = make_dir("./dataset/labels")
  # images 폴더가 없을 경우 생성
  images_path = make_dir("./dataset/images")

  make_dir("./dataset/labels/train")
  make_dir("./dataset/labels/val")
  make_dir("./dataset/labels/test")

  make_dir("./dataset/images/train")
  make_dir("./dataset/images/val")
  make_dir("./dataset/images/test")

  for img in img_list:
    img_name = img.attrib['name']
    img_path = os.path.join(folder, img_name)
    label_path = labels_path + "/" + img_name[:-4] + ".txt" ## 확장자 .jpg는 제거한 상태로 넣어줘야함
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
    else:
      try:
        shutil.move(img_path, os.path.join(images_path, img_name))
        print(f"{img_path} has been moved to images dir.")
      except:
        print(f"{img_path} already moved to images dir.")
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
        if dir in ['images', 'labels']:
          continue
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

  root_path = "/content/dataset/" ## 사용 환경에 맞게 바꿔줘야 합니다
  train_path = "images/train/"
  val_path = "images/val/"
  test_path = "images/test/"

  new_label_dict = {}
  for k, v in label_dict.items():
      new_label_dict[v] = k

  data['names'] = new_label_dict
  data['nc'] = len(label_dict)
  data['path'] = root_path
  data['train'] = train_path
  data['val'] = val_path
  data['test'] = test_path

  with open('./data.yaml', 'w') as f:
    yaml.dump(data, f)
  print("data.yaml 생성 완료!") 


#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


def holdout_split():
    images_path = "./dataset/images"
    labels_path = "./dataset/labels"

    # Read images and annotations
    images = [x[:-4] for x in os.listdir(images_path) if x[-3:] == "jpg"]
    annotations = [x[:-4] for x in os.listdir(labels_path) if x[-3:] == "txt"]

    images.sort()
    annotations.sort()
    
    solo_list = []

    for a in annotations:
      if a not in images:
        solo_list.append(a)

    for a in solo_list:
      annotations.remove(a)

    images = [os.path.join(images_path, x + ".jpg") for x in images]
    annotations = [os.path.join(labels_path, x + ".txt") for x in annotations]


    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 42)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 42)

    # Move the splits into their folders
    move_files_to_folder(train_images, './dataset/images/train')
    move_files_to_folder(val_images, './dataset/images/val/')
    move_files_to_folder(test_images, './dataset/images/test/')
    move_files_to_folder(train_annotations, './dataset/labels/train/')
    move_files_to_folder(val_annotations, './dataset/labels/val/')
    move_files_to_folder(test_annotations, './dataset/labels/test/')

    print("hold-out 완료")

    dir_path = "./dataset"

    for sub in os.listdir(dir_path):
      if sub not in ['images', 'labels']:
        shutil.rmtree(os.path.join(dir_path, sub))
    print("기존에 있던 images,labels를 제외한 폴더 삭제 완료")

    