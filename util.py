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
	return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
	stretched_voxel_coordinates = voxel_coordinates * spacing
	world_coordinates = stretched_voxel_coordinates + origin
	return world_coordinates

def normalizePlanes(npzarray):
	maxHU = 400.0
	minHU = -1000.0
	npzarray = (npzarray - minHU) / (maxHU - minHU)
	npzarray[npzarray > 1] = 1.0
	npzarray[npzarray < 0] = 0.0
	return npzarray

def ims_vis(ims):
    '''Function to display row of images'''
    fig, axes = plt.subplots(1, len(ims))
    for i, im in enumerate(ims):
        # axes[i].imshow(im, cmap='gray', origin='upper')
        axes[i].imshow(im, cmap='gray')
    plt.show()