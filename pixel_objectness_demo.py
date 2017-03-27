import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
from io import BytesIO
import os
import sys
import tempfile
import time

import server


base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../pixelobjectness')

caffe_binary = os.path.join(base_dir, '../deeplab-public/build/tools/caffe')

image_dir = os.path.join(base_dir, 'images')

input_image_file = os.path.join(image_dir, 'image.jpg')
output_mat_file = os.path.join(image_dir, 'image_blob_0.mat')

test_template_file_path = os.path.join(base_dir, 'test_template.prototxt')
test_file_path = os.path.join(base_dir, 'test.prototxt')
weight_file_path = os.path.join(base_dir, 'pixel_objectness.caffemodel')

gpu = os.environ.get('GPU')
gpu_param = ''
if (gpu is not None) and (gpu != ''):
    gpu_param = ' --gpu=' + gpu

caffe_cmd = caffe_binary + ' test --model=' + test_file_path + \
    ' --weights=' + weight_file_path + ' --iterations=1' + \
    gpu_param

max_image_size = 512

def segment_image(input_image):
    input_image = Image.open(BytesIO(input_image))
    image_rgb = np.asarray(input_image.convert('RGB'), dtype=np.uint8)
    # OpenCV uses BGR color
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = predict(image)
    if result is None:
        out_image = image
    else:
        out_image = draw_result(image, result)
    image_nd = cv2.imencode('.jpg', out_image)[1]
    return image_nd.tostring()


def init_lists_and_prototxt():
    input_list_file  = os.path.join(base_dir, 'image_list.txt')
    output_list_file = os.path.join(base_dir, 'output_list.txt')

    input_list  = open(input_list_file,'w')
    output_list = open(output_list_file,'w')

    input_list.write('/image.jpg\n')
    output_list.write('image\n')

    input_list.close()
    output_list.close()

    template_file = open(test_template_file_path).readlines()
    test_file = open(test_file_path, 'w')

    tokens = {}
    tokens['${IMAGE_DIR}'] = 'root_folder: \"' + image_dir + '\"'
    tokens['${OUTPUT_DIR}'] = 'prefix: \"' + image_dir + '/\"'

    tokens['${IMAGE_LIST}']        = 'source: \"' + input_list_file + '\"'
    tokens['${IMAGE_OUTPUT_LIST}'] = 'source: \"' + output_list_file + '\"'

    for line in template_file:
	    line = line.rstrip()

	    for key in tokens:
		    if line.find(key)!=-1:
			    line = '\t'+tokens[key]
			    break

	    test_file.write(line+'\n')

    test_file.close()


def predict(image):
    result = None
    if image.shape[0] > max_image_size or image.shape[1] > max_image_size:
        print('Error: image size is larger than ' + str(max_image_size))
    else:
        cv2.imwrite(input_image_file, image)
        
        status = os.system(caffe_cmd)

        if status == 0:
            result = sio.loadmat(output_mat_file)['data']
            result = result.reshape(result.shape[0], result.shape[1], 2)
            result = result.transpose(1, 0, 2)
    return result


def draw_result(image, result, alpha=0.5, color=(0, 255, 0)):
    rows = min(image.shape[0], result.shape[0])
    cols = min(image.shape[1], result.shape[1])
    out_image = image[0:rows, 0:cols, :]
    result = result[0:rows, 0:cols, :]
    mask = result[:, :, 1] > result[:, :, 0]
    out_image[mask] = alpha*out_image[mask] + (1-alpha)*np.array(color)
    return out_image


init_lists_and_prototxt()
server.serve(segment_image, port=os.environ.get('PORT'))

