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

dataDir = '../data/coco'
dataType = 'INTP_'
dataType_out = 'intp'
boxes_dim = 6
display_step = 500

text_path = '%s/interpolate_captions.txt'%(dataDir)
with open(text_path, "r") as f:
	caps = f.readlines()

xs_total, ys_total, ws_total, hs_total = [], [], [], []
fout_bbox_label = open('%s/bbox_label/input_%s.txt'%(dataDir, dataType_out), 'w')
fout_filename = open('%s/bbox_label/filenames_%s.txt'%(dataDir, dataType_out), 'w')

for cap_ind in xrange(0,len(caps)):
	if cap_ind % display_step == 0:
		print('%07d / %07d'%(cap_ind, len(caps)))

	cap = caps[cap_ind]
	boxes_arr = np.ones((1,6))

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

	line_new = "\t".join([" ".join(tokens_new), line])
	fout_bbox_label.write(line_new+'\n')
	fout_filename.write('%s%d\n'%(dataType, cap_ind))

fout_bbox_label.close()
fout_filename.close()
