import sys
sys.path.append('/home/zlp/dev/caffe/python')
import os
import os.path as osp
import caffe
from caffe import layers as L, params as P, to_proto

############ ############
def conv_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
    conv = L.Convolution(bottom,
    	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
    	num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
    	#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
    	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
        engine=1)
    relu = L.ReLU(conv, in_place=True, engine=1)
    return conv, relu
############ ############
def conv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, phase='train'):
	conv = L.Convolution(bottom,
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0)
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], engine=1)
	if phase == 'train':
		bn = L.BatchNorm(conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)], in_place=True)
	else:
		bn = L.BatchNorm(conv, use_global_stats=1)
	scale = L.Scale(bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0), in_place=True)
	relu = L.ReLU(scale, in_place=True, engine=1)
	return conv, bn, scale, relu
############ ############
def deconv_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
	deconv = L.Deconvolution(bottom,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
			#weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
			weight_filler=dict(type='msra'), bias_term=0,
			engine=1))
	relu = L.ReLU(deconv, in_place=True, engine=1)
	return deconv, relu
############ ############
def deconv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, phase='train'):
	deconv = L.Deconvolution(bottom,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
			weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
			engine=1))
	if phase == 'train':
		bn = L.BatchNorm(deconv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)], in_place=True)
	else:
		bn = L.BatchNorm(deconv, use_global_stats=1)
	scale = L.Scale(bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0), in_place=True)
	relu = L.ReLU(scale, in_place=True, engine=1)
	return deconv, bn, scale, relu
############ ############
def max_pool(bottom, pad=0, kernel_size=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride)
############ ############
def ave_pool(bottom, pad=0, kernel_size=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.AVE, pad=pad, kernel_size=kernel_size, stride=stride)
############ ############
def max_pool_nd(bottom, pad=[0,0,0], kernel_size=[2,2,2], stride=[2,2,2]):
	return L.PoolingND(bottom, pool=0, pad=pad, kernel_size=kernel_size, stride=stride, engine=1)
############ ############
def ave_pool_nd(bottom, pad=[0,0,0], kernel_size=[2,2,2], stride=[2,2,2]):
	return L.PoolingND(bottom, pool=1, pad=pad, kernel_size=kernel_size, stride=stride, engine=1)


# def multiscale():


def Net3DBN_S(input_dims, class_nums, ignore_label, phase="TRAIN"):
	net = caffe.NetSpec()

	############ d0 ############
	### a ###
	# net.data, net.label = L.HDF5Data(batch_size=batch_size, source=input_file, top=2, include=dict(phase=caffe.TRAIN))
	net.data = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	if phase == "TRAIN":
		input_dims = [input_dims[0], 1, 1, 1, 1]
		net.label = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	### b ###
	net.d0b_conv = L.Convolution(net.data,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=[1,3,3], kernel_size=[3,7,7], stride=2,
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d0b_bn = L.BatchNorm(net.d0b_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d0b_bn = L.BatchNorm(net.d0b_conv, in_place=True, use_global_stats=1)
	net.d0b_scale = L.Scale(net.d0b_bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d0b_relu = L.ReLU(net.d0b_scale, in_place=True, engine=1)
	# # ### c ###
	# net.d0c_conv = L.Convolution(net.d0b_scale,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=64, pad=1, kernel_size=3, stride=1, 
	# 	weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	# 	engine=1)
	# if phase == "TRAIN":
	# 	net.d0c_bn = L.BatchNorm(net.d0c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	# else:
	# 	net.d0c_bn = L.BatchNorm(net.d0c_conv, use_global_stats=1)
	# net.d0c_scale = L.Scale(net.d0c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	# net.d0c_relu = L.ReLU(net.d0c_scale, in_place=True, engine=1)

	
	############ d1 ############
	### a ### First pooling
	net.d1a_pool = L.PoolingND(net.d0b_scale,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d1b_conv = L.Convolution(net.d1a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN": 
		net.d1b_bn = L.BatchNorm(net.d1b_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1b_bn = L.BatchNorm(net.d1b_conv, in_place=True, use_global_stats=1)
	net.d1b_scale = L.Scale(net.d1b_bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1b_relu = L.ReLU(net.d1b_scale, in_place=True, engine=1)
	### c ###
	net.d1c_conv = L.Convolution(net.d1b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d1c_bn = L.BatchNorm(net.d1c_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1c_bn = L.BatchNorm(net.d1c_conv, in_place=True, use_global_stats=1)
	net.d1c_scale = L.Scale(net.d1c_bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1c_relu = L.ReLU(net.d1c_scale, in_place=True, engine=1)


	############ d2 ############
	### a ###
	net.d2a_pool = L.PoolingND(net.d1c_scale,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d2b_conv = L.Convolution(net.d2a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2b_bn = L.BatchNorm(net.d2b_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2b_bn = L.BatchNorm(net.d2b_conv, in_place=True, use_global_stats=1)
	net.d2b_scale = L.Scale(net.d2b_bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2b_relu = L.ReLU(net.d2b_scale, in_place=True, engine=1)
	### c ###
	net.d2c_conv = L.Convolution(net.d2b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2c_bn = L.BatchNorm(net.d2c_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2c_bn = L.BatchNorm(net.d2c_conv, in_place=True, use_global_stats=1)
	net.d2c_scale = L.Scale(net.d2c_bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2c_relu = L.ReLU(net.d2c_scale, in_place=True, engine=1)


	############ d3 ############
	### a ### Third Pooling
	net.d3a_pool = L.PoolingND(net.d2c_scale,
		pool=1,
		kernel_size=[3,5,5],
		stride=1,
		engine=1)

	net.d3b_ip = L.InnerProduct(net.d3a_pool, num_output=class_nums)
	# net.d3c_ip = L.InnerProduct(net.d3a_pool, num_output=2)

	############ Loss ############
	if phase == "TRAIN":
		net.loss = L.SoftmaxWithLoss(net.d3b_ip, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
	else:
		net.prob = L.Softmax(net.d3b_ip,
			phase=1)

	return net.to_proto()

def Net3DBN(input_dims, class_nums, ignore_label, phase="TRAIN"):
	net = caffe.NetSpec()

	############ d0 ############
	### a ###
	# net.data, net.label = L.HDF5Data(batch_size=batch_size, source=input_file, top=2, include=dict(phase=caffe.TRAIN))
	net.data = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	if phase == "TRAIN":
		input_dims = [input_dims[0], 1, 1, 1, 1]
		net.label = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	### b ###
	net.d0b_conv = L.Convolution(net.data,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=[1,3,3], kernel_size=[3,7,7], stride=2,
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d0b_bn = L.BatchNorm(net.d0b_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d0b_bn = L.BatchNorm(net.d0b_conv, in_place=True, use_global_stats=1)
	net.d0b_scale = L.Scale(net.d0b_conv, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d0b_relu = L.ReLU(net.d0b_conv, in_place=True, engine=1)
	# ### c ###
	net.d0c_conv = L.Convolution(net.d0b_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d0c_bn = L.BatchNorm(net.d0c_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d0c_bn = L.BatchNorm(net.d0c_conv, in_place=True, use_global_stats=1)
	net.d0c_scale = L.Scale(net.d0c_conv, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d0c_relu = L.ReLU(net.d0c_conv, in_place=True, engine=1)

	
	############ d1 ############
	### a ### First pooling
	net.d1a_pool = L.PoolingND(net.d0c_conv,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d1b_conv = L.Convolution(net.d1a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN": 
		net.d1b_bn = L.BatchNorm(net.d1b_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1b_bn = L.BatchNorm(net.d1b_conv, in_place=True, use_global_stats=1)
	net.d1b_scale = L.Scale(net.d1b_conv, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1b_relu = L.ReLU(net.d1b_conv, in_place=True, engine=1)
	### c ###
	net.d1c_conv = L.Convolution(net.d1b_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d1c_bn = L.BatchNorm(net.d1c_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1c_bn = L.BatchNorm(net.d1c_conv, in_place=True, use_global_stats=1)
	net.d1c_scale = L.Scale(net.d1c_conv, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1c_relu = L.ReLU(net.d1c_conv, in_place=True, engine=1)


	############ d2 ############
	### a ###
	net.d2a_pool = L.PoolingND(net.d1c_conv,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d2b_conv = L.Convolution(net.d2a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2b_bn = L.BatchNorm(net.d2b_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2b_bn = L.BatchNorm(net.d2b_conv, in_place=True, use_global_stats=1)
	net.d2b_scale = L.Scale(net.d2b_conv, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2b_relu = L.ReLU(net.d2b_conv, in_place=True, engine=1)
	### c ###
	net.d2c_conv = L.Convolution(net.d2b_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2c_bn = L.BatchNorm(net.d2c_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2c_bn = L.BatchNorm(net.d2c_conv, in_place=True, use_global_stats=1)
	net.d2c_scale = L.Scale(net.d2c_conv, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2c_relu = L.ReLU(net.d2c_conv, in_place=True, engine=1)


	############ d3 ############
	### a ### Third Pooling
	net.d3a_pool = L.PoolingND(net.d2c_conv,
		pool=1,
		kernel_size=[3,5,5],
		stride=1,
		engine=1)
	### b ###
	net.d3b_conv = L.Convolution(net.d3a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=512, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d3b_bn = L.BatchNorm(net.d3b_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d3b_bn = L.BatchNorm(net.d3b_conv, in_place=True, use_global_stats=1)
	net.d3b_scale = L.Scale(net.d3b_conv, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d3b_relu = L.ReLU(net.d3b_conv, in_place=True, engine=1)
	### c ###
	net.d3c_conv = L.Convolution(net.d3b_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=512, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d3c_bn = L.BatchNorm(net.d3c_conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d3c_bn = L.BatchNorm(net.d3c_conv, in_place=True, use_global_stats=1)
	net.d3c_scale = L.Scale(net.d3c_conv, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d3c_relu = L.ReLU(net.d3c_conv, in_place=True, engine=1)


	net.d3b_ip = L.InnerProduct(net.d3c_conv, num_output=class_nums)
	# net.d3c_ip = L.InnerProduct(net.d3a_pool, num_output=2)

	############ Loss ############
	if phase == "TRAIN":
		net.loss = L.SoftmaxWithLoss(net.d3b_ip, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
	else:
		net.prob = L.Softmax(net.d3b_ip,
			phase=1)

	return net.to_proto()

####################################################
# batch channel z y x
input_dims = [384, 1, 24, 40, 40]
class_nums = 2
ignore_label = 255
net_name = 'Net3DBN'
#net_name = 'Net3DBN_S'

dirname = '{}'.format(net_name)
prototxt_dir = osp.abspath(osp.join('Prototxt', dirname))

if not osp.exists(prototxt_dir):
	os.makedirs(prototxt_dir)
prototxt_train = osp.abspath(osp.join(prototxt_dir, 'train.prototxt'))
prototxt_test = osp.abspath(osp.join(prototxt_dir, 'test.prototxt'))

with open(prototxt_train, 'w') as f:
	if net_name == 'Net3DBN':
		net = Net3DBN(input_dims, class_nums, ignore_label, phase="TRAIN")
	elif net_name == 'Net3DBN_S':
		net = Net3DBN_S(input_dims, class_nums, ignore_label, phase="TRAIN")
	else:
		print 'net name error'
	f.write(str(net))

with open(prototxt_test, 'w') as f:
	if net_name == 'Net3DBN':
		net = Net3DBN(input_dims, class_nums, ignore_label, phase="TEST")
	elif net_name == 'Net3DBN_S':
		net = Net3DBN_S(input_dims, class_nums, ignore_label, phase="TEST")
	else:
		print 'net name error'
	f.write(str(net))