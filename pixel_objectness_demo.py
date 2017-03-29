import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
from io import BytesIO
import os
import sys
import tempfile
import time
import caffe
import server

base_dir = os.path.dirname(os.path.realpath(__file__))
prototxt_file =  os.path.join(base_dir, 'pixel_objectness_demo.prototxt')
weight_file = os.path.join(base_dir, '../pixelobjectness/pixel_objectness.caffemodel')

IMAGE_SIZE = 513

net = None


def init_net():
    net = caffe.Net(prototxt_file, weight_file)
    gpu = os.environ.get('GPU')
    if (gpu is not None) and (gpu != ''):
        net.set_mode_gpu()
        gpu_device = int(gpu)
        net.set_device(gpu_device)
        print('Use GPU ' + str(gpu_device))
    else:
        net.set_mode_cpu()
        print('Use CPU')
    return net


def forward_net(net, image_bgr):
    mean = np.array([[[104.008, 116.669, 122.675]]], dtype=np.float32)
    image = image_bgr - mean

    pad_h = IMAGE_SIZE - image.shape[0]
    pad_w = IMAGE_SIZE - image.shape[1]
    image = np.pad(image, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    ch = image.shape[2]
    input_blob = image.transpose(2, 0, 1).reshape((1, ch, IMAGE_SIZE, IMAGE_SIZE))

    output = net.forward_all(**{net.inputs[0]: input_blob})
    output_blob = output[net.outputs[0]]
    return output_blob[0].transpose(1, 2, 0)


def segment_image(input_image, return_mask=False):
    input_image = Image.open(BytesIO(input_image))
    image_rgb = np.asarray(input_image.convert('RGB'), dtype=np.uint8)
    # OpenCV uses BGR color
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = predict(image)
    if result is None:
        out_image = image
    else:
        if return_mask:
            out_image = draw_mask(image, result)
        else:
            out_image = draw_result(image, result)
    image_nd = cv2.imencode('.jpg', out_image)[1]
    return image_nd.tostring()


def predict(image):
    result = None
    if image.shape[0] > IMAGE_SIZE or image.shape[1] > IMAGE_SIZE:
        print('Error: image size is larger than ' + str(IMAGE_SIZE))
    else:
        result = forward_net(net, image)
    return result


def draw_result(image, result, alpha=0.5, color=(0, 255, 0)):
    rows = min(image.shape[0], result.shape[0])
    cols = min(image.shape[1], result.shape[1])
    out_image = image[0:rows, 0:cols, :]
    result = result[0:rows, 0:cols, :]
    mask = result[:, :, 1] > result[:, :, 0]
    out_image[mask] = alpha*out_image[mask] + (1-alpha)*np.array(color)
    return out_image
    

def draw_mask(image, result):
    rows = min(image.shape[0], result.shape[0])
    cols = min(image.shape[1], result.shape[1])
    result = result[0:rows, 0:cols, :]
    mask = 255*(result[:, :, 1] > result[:, :, 0]).astype(np.uint8)
    return mask


net = init_net()
server.serve(segment_image, port=os.environ.get('PORT'))
