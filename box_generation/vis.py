from glob import glob
import pandas as pd
import sys
import ntpath
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nltk.tokenize import RegexpTokenizer
from skimage.transform import resize
import os
import errno

checkpoint = '2018_10_23_14_55_12'
gen_dir = '../data/coco/gen_masks_%s/*'%(checkpoint)
gt_dir = '../data/coco/masks/'
img_dir = '../data/coco/images/'
txt_dir = '../data/coco/text/'
output_dir = '../vis_bboxes_%s/'%(checkpoint)
CAPS_PER_IMG = 5
FONT_MAX = 40
FONT_REAL = 30
MAX_WORD_NUM = 20
FNT = ImageFont.truetype('../data/coco/share/Pillow/Tests/fonts/FreeMono.ttf', FONT_REAL)
STD_IMG_SIZE = 256
VIS_SIZE = STD_IMG_SIZE
OFFSET = 2
SHOW_LIMIT = 500

def path_leaf(path):
	return ntpath.basename(path)

def load_captions(cap_path):
    all_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().decode('utf8').split('\n')
        cnt = 0
        for cap in captions:
            if len(cap) == 0:
                continue
            cap = cap.replace("\ufffd\ufffd", " ")
            # picks out sequences of alphanumeric characters as tokens
            # and drops everything else
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(cap.lower())
            # print('tokens', tokens)
            if len(tokens) == 0:
                print('cap', cap)
                continue

            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(t)
            sentence = ' '.join(tokens_new)
            all_captions.append(sentence)
            cnt += 1
            if cnt == CAPS_PER_IMG:
                break
        if cnt < CAPS_PER_IMG:
            print('ERROR: the captions for %s less than %d'
                  % (cap_path, cnt))
    return all_captions

def draw_plate(bboxes):
	bbox_plate = Image.fromarray((np.ones((VIS_SIZE, VIS_SIZE, 3))*255).astype(np.uint8))
	if bboxes is None:
		return bbox_plate

	d = ImageDraw.Draw(bbox_plate)
	for i in xrange(bboxes.shape[0]):
		left, top, width, height, label = bboxes[i, :5]
		label = int(label)
		color = (210-label*2,label*3,50+label*2)
		d.rectangle([left, top, left+width-1, top+height-1], outline=color)
		d.text([left+5, top+5], str(label), fill=color)
	del d
	return bbox_plate

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def is_non_zero_file(fpath):  
    return True if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else False

mkdir_p(output_dir)

gen_paths = glob(gen_dir)
keys = [path_leaf(gen_path) for gen_path in gen_paths]

count = 0
for key in keys:
	if count >= SHOW_LIMIT:
		break

	# 1. load image
	img_path = '%s%s.jpg'%(img_dir, key)
	img = np.array(Image.open(img_path))
	img_height, img_width = img.shape[0], img.shape[1]
	height_scale = STD_IMG_SIZE/float(img_height)
	width_scale = STD_IMG_SIZE/float(img_width)
	img = resize(img, [STD_IMG_SIZE, STD_IMG_SIZE])
	img = Image.fromarray((img*255).astype(np.uint8))

	# 2. load captions
	cap_path = '%s%s.txt'%(txt_dir, key)
	captions = load_captions(cap_path)

	# 3. load gt bboxes
	gt_bbox_path = '%s%s/boxes.txt'%(gt_dir, key)
	if is_non_zero_file(gt_bbox_path):
		gt_boxes = pd.read_csv(gt_bbox_path, header=None).astype(int)
		gt_boxes = np.array(gt_boxes)
		gt_boxes[:,[0,2]] = gt_boxes[:,[0,2]]*width_scale
		gt_boxes[:,[1,3]] = gt_boxes[:,[1,3]]* height_scale
	else:
		gt_boxes = None

	# 4. load gen bboxes
	gen_bbox_paths = glob('%s%s/*'%(gen_dir, key))
	gen_bbox_paths_indices = [int(path_leaf(gen_bbox_path)) for gen_bbox_path in gen_bbox_paths]
	gen_bbox_paths_indices = np.argsort(gen_bbox_paths_indices)
	gen_bbox_paths = [gen_bbox_paths[index] for index in gen_bbox_paths_indices]
	gen_boxes_set = []
	for gen_bbox_path in gen_bbox_paths:
		sub_gen_bbox_path = '%s/boxes.txt'%(gen_bbox_path)
		if is_non_zero_file(sub_gen_bbox_path):
			gen_boxes = pd.read_csv(sub_gen_bbox_path, header=None).astype(int)
			gen_boxes = np.array(gen_boxes)
			gen_boxes_set.append(gen_boxes)
		else:
			gen_boxes_set.append(None)

	# 5. draw text
	text_convas = np.ones([VIS_SIZE + CAPS_PER_IMG * FONT_MAX,  7 * VIS_SIZE, 3], dtype=np.uint8)
	img_plate = Image.fromarray(text_convas)

	d = ImageDraw.Draw(img_plate)
	for i in xrange(len(captions)):
		d.text((2, i*FONT_REAL), captions[i], font=FNT, fill=(255, 255, 255, 255))

	del d

	# 6. draw images
	starting_y = CAPS_PER_IMG * FONT_MAX
	starting_x = VIS_SIZE+OFFSET
	
	img_plate.paste(img, (0, starting_y))

	# 7. draw gt boxes
	gt_bbox_plate = draw_plate(gt_boxes)
	img_plate.paste(gt_bbox_plate, (starting_x, starting_y))

	# 8. draw gen boxes
	for i in xrange(len(gen_boxes_set)):
		gen_boxes = gen_boxes_set[i]
		gen_bbox_plate = draw_plate(gen_boxes)
		img_plate.paste(gen_bbox_plate, (starting_x*(i+2), starting_y))

	img_plate.save('%s%s.jpg'%(output_dir, key))