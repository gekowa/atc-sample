from caffe import layers as L
from caffe import params as P
import caffe
import numpy as np
import time

pt_file = "caffe_test.pbtxt"
cm_file = "caffe_test.caffemodel"


def lenet():
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n["data"] = L.Input(shape=[dict(dim=[1, 3, 1, 1])], ntop=1)
    n.conv1 = L.Convolution(n.data, kernel_size=1, num_output=4, weight_filler=dict(type='gaussian'))
    n.reshape1a = L.Reshape(n.conv1, reshape_param={'shape':{'dim':0, 'dim':-1}})
    n.conv2 = L.Convolution(n.data, kernel_size=1, num_output=4, weight_filler=dict(type='gaussian'))
    n.reshape2a = L.Reshape(n.conv2, reshape_param={'shape':{'dim':0 ,'dim':-1}})

    n.concat3 = L.Concat(n.reshape1a,n.reshape2a,concat_param = dict(axis=-1))

    return n.to_proto()


def save_model():
    with open(pt_file, 'w') as f:
        f.write(str(lenet()))

    net = caffe.Net(pt_file, caffe.TEST)
    net.save(cm_file)

def caffe_forward():

    input = np.random.randn(1, 3, 1, 1).astype(np.float32)
    net = caffe.Net(pt_file, cm_file, caffe.TEST)
    net.blobs['data'].data[...] = input
    out = net.forward()
    out_data_1 = out['concat3'].astype(np.float32)
    out_data_1.tofile("caffe_cpu_output.bin")
    input.tofile("input.bin")

if __name__ == "__main__":
    save_model()
    time.sleep(1)
    caffe_forward()
