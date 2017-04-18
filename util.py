import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import csv
import scipy
from scipy import ndimage

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, 
origin and spacing of the image.
'''
def load_itk_image(filename):
	# Reads the image using SimpleITK
	itkimage = sitk.ReadImage(filename)
	# Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
	ct_scan = sitk.GetArrayFromImage(itkimage)
	# While in numpy, an array is indexed in the opposite order (z,y,x).
	# Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
	origin = np.array(list(reversed(itkimage.GetOrigin())))
	# Read the spacing along each dimension
	spacing = np.array(list(reversed(itkimage.GetSpacing())))

	return ct_scan, origin, spacing

def read_csv(filename):
	lines = []
	with open(filename, 'rb') as f:
		scvreader = csv.reader(f)
		for line in scvreader:
			lines.append(line)

	return lines

'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
	stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
	voxel_coordinates = stretched_voxel_coordinates / spacing
	# transform coordinates to int
	voxel_coordinates = np.round(voxel_coordinates).astype(np.int)
	return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
	stretched_voxel_coordinates = voxel_coordinates * spacing
	world_coordinates = stretched_voxel_coordinates + origin
	return world_coordinates

def hu_normalize(npzarray, hu_window):
	'''Houndsfield Unit
	and Normalize'''
	minHU = hu_window[0]
	maxHU = hu_window[1]
	npzarray = (npzarray - minHU) / (maxHU - minHU)
	npzarray[npzarray > 1] = 1.0
	npzarray[npzarray < 0] = 0.0
	return npzarray

def hounsfield_unit_window(im, hu_window):
    """ The Hounsfield unit values will be windowed in the range
    [min_bound, max_bound] to exclude irrelevant organs and objects.
    """
    hu_window = np.array(hu_window, np.float32)

    im[im<hu_window[0]] = hu_window[0]
    im[im>hu_window[1]] = hu_window[1]
    return im

def normalizer(im, src_range, dst_range=[0.0, 1.0]):
    # Normalize src_range to dst_range
    src_range = np.array(src_range, np.float32)
    dst_range = np.array(dst_range, np.float32)
    
    im = (im - src_range[0]) / (src_range[1] - src_range[0])
    im = im * (dst_range[1] - dst_range[0]) + dst_range[0]
    im[im<dst_range[0]] = dst_range[0]
    im[im>dst_range[1]] = dst_range[1]
    return im

def ims_vis(ims):
    '''Function to display row of images'''
    fig, axes = plt.subplots(1, len(ims))
    for i, im in enumerate(ims):
        # axes[i].imshow(im, cmap='gray', origin='upper')
        axes[i].imshow(im, cmap='gray')
    plt.show()

def vis_seg(axes, ims):
	'''Function to display row of images'''
	for i, im in enumerate(ims):
		axes[i].imshow(im, cmap='gray', origin='upper')
	plt.show()
	plt.pause(0.00001)

def crop_patch(image_shape, patch_shape, centre_coordinates):
	'''
	Crop patch from image based on patch_shape and centre_coordinates
	Return patch_b which is the begin index of the patch in original image
	patch can construct from im[patch_b:patch_b+patch_shape]
	'''
	im_shape = np.asarray(image_shape)
	pa_shape = np.asarray(patch_shape)
	centre = np.asarray(centre_coordinates)
	assert len(im_shape)==len(pa_shape) and  len(pa_shape)==len(centre), 'The dimensions must be the same for image_shape, patch_shape, centre_coordinates'

	patch_b = np.round(centre - pa_shape/2).astype(np.int)
	# Prevent patch_b index out of boundary
	patch_b[patch_b<0] = 0
	# Prevent patch_e index out of boundary
	index_out = (patch_b + pa_shape - im_shape) > 0
	patch_b[index_out] = im_shape[index_out] - pa_shape[index_out]

	return patch_b
