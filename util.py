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

# def ims_vis(ims):
#     '''Function to display row of images'''
#     fig, axes = plt.subplots(1, len(ims))
#     for i, im in enumerate(ims):
#         # axes[i].imshow(im, cmap='gray', origin='upper')
#         axes[i].imshow(im, cmap='gray')
#     plt.show()

def ims_vis(axes, ims):
	'''Function to display row of images'''
	for i, im in enumerate(ims):
		axes[i].imshow(im, cmap='gray', origin='upper')
	plt.show()
	plt.pause(0.00001)

def crop_patch(image, patch_size, centre_coordinates):
	'''
	Crop patch from image based on patch_size and centre_coordinates
	Return cropped_patch
	'''
	im_shape = np.asarray(image.shape)
	patch_shape = np.asarray(patch_size)
	centre_coordi = np.asarray(centre_coordinates)
	assert len(im_shape)==len(patch_shape) and  len(patch_shape)==len(centre_coordi), 'The dimensions must be the same for image, patch_size, centre_coordinates'
	assert np.sum((patch_shape - im_shape) > 0) == 0, 'patch size must be not greater than image size'

	patch_start = np.round(centre_coordi - patch_shape/2).astype(np.int)
	# Prevent patch_start index out of boundary
	patch_start[patch_start<0] = 0
	# Prevent patch_end index out of boundary
	index_out = (patch_start + patch_shape - im_shape) > 0
	patch_start[index_out] = im_shape[index_out] - patch_shape[index_out]
	patch_end = patch_start + patch_shape

	cropped_patch = image[patch_start[0]:patch_end[0], patch_start[1]:patch_end[1], patch_start[2]:patch_end[2]]

	return cropped_patch

def im_list_to_blob(ims, Dtype=np.float32):
    """Convert a list of images(BGR) into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    assert len(max_shape) == 3, 'image must have 3 dimensions'
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]), dtype=Dtype)
    for i in xrange(num_images):
        im = ims[i].astype(dtype=Dtype, copy=False)
        blob[i, 0:im.shape[0], 0:im.shape[1], 0:im.shape[2]] = im

    blob = blob[np.newaxis, ...]
    channel_swap = (1, 0, 2, 3, 4)
    blob = blob.transpose(channel_swap)
    return blob
