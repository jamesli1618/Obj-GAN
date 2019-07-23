import numpy as np
import skimage.io as io
import sys
from PIL import Image
import glob
import ntpath
import os
from nltk.tokenize import RegexpTokenizer

def read_hmap(hmap_path):
    hmap = Image.open(hmap_path)
    hmap = np.asarray(hmap)
    hmap = np.squeeze(hmap[:,:,0])
    return hmap

x_index, y_index, w_index, h_index, l_index, c_index = 0, 1, 2, 3, 4, 5

def path_leaf(path):
	return ntpath.basename(path)

def calc_sort_size(boxes_arr):
	# boxes_arr (type = numpy array): boxes_num x 6 (x, y, w, h, l, crowd_l)
	# calculate the product of width and height
	sizes = np.multiply(boxes_arr[:, w_index], boxes_arr[:, h_index])

	# sort sizes in the ascending order
	sorted_indices = np.argsort(sizes)[::-1].tolist()
	sorted_boxes_arr = boxes_arr[sorted_indices,:]

	return sorted_boxes_arr

def reorg_boxes_arr(boxes_arr):
	# new_boxes_arr (type = numpy array): boxes_num x 6 (centern_x, center_y, w, h/w, l, crowd_l)
	new_boxes_arr = np.zeros(boxes_arr.shape)

	# centern_x
	new_boxes_arr[:,0] = boxes_arr[:,x_index]+boxes_arr[:,w_index]/2.0
	# centern_y
	new_boxes_arr[:,1] = boxes_arr[:,y_index]+boxes_arr[:,h_index]/2.0
	# h
	new_boxes_arr[:,3] = np.divide(boxes_arr[:,h_index], boxes_arr[:,w_index])
	# w, l
	new_boxes_arr[:,[2,4]] = boxes_arr[:,[w_index,l_index]]

	return new_boxes_arr

# COCO_train2014_

dataDir = '../data/coco'
dataType = 'COCO_train2014_'
dataType_out = 'train2014'
outputDirname = 'bbox_label'

textDir = '%s/text/%s*.txt'%(dataDir, dataType)
mainBoxDir = '%s/masks'%(dataDir)

text_files = glob.glob(textDir)
text_nms = [path_leaf(path) for path in text_files]
text_nms_wo_ext = [name.split(".")[0] for name in text_nms]

max_bbox_num = 10
boxes_dim = 6
std_img_size = 256
use_crowd = False
display_step = 500

xs_total, ys_total, ws_total, hs_total = [], [], [], []

fout_filename = open('%s/%s/filenames_%s.txt'%(dataDir, outputDirname, dataType_out), 'w')
fout_bbox_label = open('%s/%s/input_%s.txt'%(dataDir, outputDirname, dataType_out), 'w')
fout_mean_std = open('%s/%s/mean_std_%s.txt'%(dataDir, outputDirname, dataType_out), 'w')

for img_ind in xrange(0,len(text_nms_wo_ext)):
	if img_ind % display_step == 0:
		print('%07d / %07d'%(img_ind, len(text_nms_wo_ext)))

	with open(text_files[img_ind], "r") as f:
		caps = f.readlines()

	hmapDir = '%s/%s/*.jpg'%(mainBoxDir, text_nms_wo_ext[img_ind])
	hmap_files = glob.glob(hmapDir)

	# check if hmap_files exist
	if len(hmap_files) == 0:
		continue

	boxFile = '%s/%s/boxes.txt'%(mainBoxDir, text_nms_wo_ext[img_ind])
	with open(boxFile, "r") as f:
		boxes = f.readlines()

	boxes_arr = np.zeros((len(boxes), boxes_dim), dtype=float)
	noncrowd_mask = []
	for box_ind in xrange(len(boxes)):
		box = boxes[box_ind].split(',')
		box = [int(float(num)) for num in box]
		boxes_arr[box_ind,:] = np.array(box)
		if box[c_index] == 0:
			noncrowd_mask.append(box_ind)

	if not use_crowd and len(noncrowd_mask) > 0:
		boxes_arr = boxes_arr[noncrowd_mask, :]
	elif not use_crowd and len(noncrowd_mask) == 0:
		continue

	assert np.sum(boxes_arr[:,c_index]) == 0

	hmap = read_hmap(hmap_files[0])
	height, width = hmap.shape[0], hmap.shape[1]

	height_scale = std_img_size/float(height)
	width_scale = std_img_size/float(width)

	boxes_arr[:, x_index] = boxes_arr[:, x_index]*width_scale
	boxes_arr[:, x_index] = np.clip(boxes_arr[:, x_index], 1, std_img_size)
	boxes_arr[:, y_index] = boxes_arr[:, y_index]*height_scale
	boxes_arr[:, y_index] = np.clip(boxes_arr[:, y_index], 1, std_img_size)
	boxes_arr[:, w_index] = boxes_arr[:, w_index]*width_scale
	boxes_arr[:, w_index] = np.clip(boxes_arr[:, w_index], 1, std_img_size)
	boxes_arr[:, h_index] = boxes_arr[:, h_index]*height_scale
	boxes_arr[:, h_index] = np.clip(boxes_arr[:, h_index], 1, std_img_size)

	# calculate and sort the distance between each bbox and image's bottom center point
	boxes_arr = calc_sort_size(boxes_arr)
	tmp_max_bbox_num = min(max_bbox_num, boxes_arr.shape[0])
	boxes_arr = boxes_arr[:max_bbox_num,:]
	boxes_arr = reorg_boxes_arr(boxes_arr)

	xs_total = np.concatenate((xs_total, boxes_arr[:,x_index]))
	ys_total = np.concatenate((ys_total, boxes_arr[:,y_index]))
	ws_total = np.concatenate((ws_total, boxes_arr[:,w_index]))
	hs_total = np.concatenate((hs_total, boxes_arr[:,h_index]))

	xs = boxes_arr[:, x_index].tolist()
	xs = ['%.2f'%(x) for x in xs]
	ys = boxes_arr[:, y_index].tolist()
	ys = ['%.2f'%(y) for y in ys]
	ws = boxes_arr[:, w_index].tolist()
	ws = ['%.2f'%(w) for w in ws]
	hs = boxes_arr[:, h_index].tolist()
	hs = ['%.2f'%(h) for h in hs]
	ls = boxes_arr[:, l_index].tolist()
	ls = [str(int(l)) for l in ls]

	line = "\t".join([" ".join(xs), " ".join(ys), " ".join(ws), " ".join(hs), " ".join(ls)])

	real_caps_num = 0
	for cap_ind in xrange(len(caps)):
		cap = caps[cap_ind].decode('utf8').split(',')[0].split('\n')[0].replace("\ufffd\ufffd", " ")
		tokenizer = RegexpTokenizer(r'\w+')
		tokens = tokenizer.tokenize(cap.lower())
		tokens_new = []
		for t in tokens:
			t = t.encode('ascii', 'ignore').decode('ascii')
			if len(t) > 0:
				tokens_new.append(t)

		if len(tokens_new) == 0:
			continue

		real_caps_num += 1
		line_new = "\t".join([" ".join(tokens_new), line])
		fout_bbox_label.write(line_new+'\n')

		fout_filename.write(text_nms_wo_ext[img_ind]+'\n')

fout_filename.close()
fout_bbox_label.close()

xs_mean, xs_std = np.mean(xs_total), np.std(xs_total)
ys_mean, ys_std = np.mean(ys_total), np.std(ys_total)
ws_mean, ws_std = np.mean(ws_total), np.std(ws_total)
hs_mean, hs_std = np.mean(hs_total), np.std(hs_total)

fout_mean_std.write('%f %f\n'%(xs_mean, xs_std))
fout_mean_std.write('%f %f\n'%(ys_mean, ys_std))
fout_mean_std.write('%f %f\n'%(ws_mean, ws_std))
fout_mean_std.write('%f %f\n'%(hs_mean, hs_std))

fout_mean_std.close()
