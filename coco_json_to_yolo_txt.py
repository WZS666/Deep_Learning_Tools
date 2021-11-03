import os
import json
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='./Bbox/coco/xxx.json')
parser.add_argument('--save_path', default='./Bbox/yolo/',type=str)
arg = parser.parse_args()

def convert(size,category_id, box):
	result = ''
	for i in range(len(box)):
		x, y, w, h = box[i]
		ct_id = category_id[i]
		width = size[0]
		height = size[1]
		x = str(x / width)
		y = str(y / height)
		w = str(w / width)
		h = str(h / height)
		result += str(ct_id) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
	return str(result)

if __name__ == '__main__':
	json_file = arg.json_path
	ana_txt_save_path = arg.save_path

	data = json.load(open(json_file, 'r',encoding="utf-8"))
	if not os.path.exists(ana_txt_save_path):
		os.makedirs(ana_txt_save_path)
	# print(json_file.split("/"))
	folder_name = json_file.split("/")[3]+"/"+json_file.split("/")[4]
	if not os.path.exists(ana_txt_save_path+"/"+folder_name):
		os.makedirs(ana_txt_save_path+"/"+folder_name)

	id_map = {}
	with open(os.path.join(ana_txt_save_path, json_file.split("/")[3], 'class.txt'), 'w', encoding="utf-8") as f:
		for i, category in enumerate(data['categories']):
			f.write(f"{category['name']}\n")
			id_map[category['id']] = i
	list_file = open(os.path.join(ana_txt_save_path, json_file.split("/")[3], 'text.txt'), 'w', encoding="utf-8")
	for img in tqdm(data['images']):
		filename = img["file_name"]
		img_width = 1920
		img_height = 1080
		img_id = img["id"]
		head, tail = os.path.splitext(filename)
		ana_txt_name = head + ".txt"
		f_txt = open(os.path.join(ana_txt_save_path, json_file.split("/")[3], ana_txt_name), 'w', encoding="utf-8")
		for ann in data['annotations']:
			if ann['image_id'] == img_id:
				result = convert((img_width, img_height), ann['category_id'], ann["bbox"])
				f_txt.write(result)
		f_txt.close()

		list_file.write('./images/train/%s.jpg\n' %(head))
	list_file.close()