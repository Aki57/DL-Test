import numpy as np
import sys
import caffe
from skimage import io

def vis_square(data):
    # 这段代码是Caffe官方完成的例子
    data = (data - data.min()) / (data.max() - data.min())
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]), (0, 1), (0, 1)) + 
        ((0, 0), ) * ( data.ndim - 3))
    data = np.pad(data, padding, mode='constant', constant_values=1)

    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data

def predict(net, transformer, img):
    input_data = np.array(img)
    input_data = input_data.reshape(1, 28, 28, 1)
    net.blobs['data'].data[...] = transformer.preprocess('data',input_data[0])
    out = net.forward()

def process(model_path, weight_path, img_path):
    net = caffe.Net(model_path, weight_path, caffe.TEST)
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    
    img = caffe.io.load_image(img_path, color=False)
    predict(net, transformer, img)
    for key in net.blobs:
        data = net.blobs[key].data
        # 只可视化全连接层以上的结果
        if data.ndim == 4:
            vis = vis_square(data[0])
            io.imsave(key + '.png', vis)

if __name__ == '__main__':
    # 这里跳过参数校验
    model_path = sys.argv[1]
    weight_path = sys.argv[2]
    img_path = sys.argv[3]
    process(model_path, weight_path, img_path)