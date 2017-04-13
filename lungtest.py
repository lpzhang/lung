import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
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
	# SimpleITK and numpy indexing access is in opposite order!
	ct_scan = sitk.GetArrayFromImage(itkimage)
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
        axes[i].imshow(im, cmap='gray', origin='upper')
    plt.show()

def seq(start, stop, step=1):
	n = int(round((stop - start)/float(step)))
	if n > 1:
		return([start + step*i for i in range(n+1)])
	else:
		return([])

'''
This function is used to create spherical regions in binary masks
at the given locations and radius.
'''
def draw_circles(image,cands,origin,spacing):
	#make empty matrix, which will be filled with the mask
	RESIZE_SPACING = spacing
	image_mask = np.zeros(image.shape)
	#run over all the nodules in the lungs
	for cand in cands:
		#get middel x-,y-, and z-worldcoordinate of the nodule
		# radius = np.ceil(cand[4])/2
		coord_x = float(cand[1])
		coord_y = float(cand[2])
		coord_z = float(cand[3])
		radius = int(np.ceil(float(cand[4])/2))

		world_coord = np.array((coord_z,coord_y,coord_x))
		#determine voxel coordinate given the worldcoordinate
		image_coord = world_2_voxel(world_coord,origin,spacing)
		print world_coord, image_coord

		#determine the range of the nodule
		noduleRange = seq(-radius, radius, RESIZE_SPACING[0])
		print noduleRange

		#create the mask
		for x in noduleRange:
			for y in noduleRange:
				for z in noduleRange:
					coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
					# if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) <= radius:
					image_mask[int(np.round(coords[0])),int(np.round(coords[1])),int(np.round(coords[2]))] = int(1)
	
	return image_mask


img_path = '/home/zlp/data/lpzhang/LungNodule2016/lungnodule/subset5/1.3.6.1.4.1.14519.5.2.1.6279.6001.504845428620607044098514803031.mhd'
cand_path = '/home/zlp/data/lpzhang/LungNodule2016/lungnodule/CSVFILES/annotations.csv'
numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
print numpyImage.shape, numpyOrigin, numpySpacing

cands = read_csv(cand_path)
cands_test = list()
print len(cands)
for cand in cands[1:]:
	# print cand[0]
	if cand[0] == '1.3.6.1.4.1.14519.5.2.1.6279.6001.504845428620607044098514803031':
		print cand
		cands_test.append(cand)
# 	if float(cand[4]) < 4:
# 	# 	pass
# 		print cand
# # for cand in cands[1:5]:
# 	worldCoord = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
# 	voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
# 	voxelWidth = 10
# 	patch_z = voxelCoord[0]
# 	patch = numpyImage[int(voxelCoord[0]), int(voxelCoord[1]-voxelWidth/2):int(voxelCoord[1]+voxelWidth/2), int(voxelCoord[2]-voxelWidth/2):int(voxelCoord[2]+voxelWidth/2)]
# 	patch = normalizePlanes(patch)
# 	print 'data'
# 	print worldCoord
# 	print voxelCoord
# 	print patch
# 	plt.imshow(patch, cmap='gray')
# 	plt.show()

# exit()
'''
This function takes the path to a '.mhd' file as input and 
is used to create the nodule masks and segmented lungs after 
rescaling to 1mm size in all directions. It saved them in the .npz
format. It also takes the list of nodule locations in that CT Scan as 
input.
'''
def create_nodule_mask(imagePath, cands):
	#if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
	img, origin, spacing = load_itk_image(imagePath)
	newimg = img.astype(dtype=np.float, copy=True)
	ims_vis([img[0,:,:],newimg[0,:,:]])
	newimg = newimg.transpose((2,1,0))
	ims_vis([img[:,:,0], newimg[:,:,0]])
	print img.shape, newimg.shape
	exit()

	#calculate resize factor
	RESIZE_SPACING = [1, 1, 1]
	# RESIZE_SPACING = [1, 0.5, 0.5]
	resize_factor = spacing / RESIZE_SPACING
	new_real_shape = img.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize = new_shape / img.shape
	new_spacing = spacing / real_resize

	print spacing, new_spacing

	#resize image
	lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)
	print lung_img.shape

	# lung_img = img

	# Segment the lung structure
	lung_img = lung_img + 1024
	# lung_mask = segment_lung_from_ct_scan(lung_img)
	lung_img = lung_img - 1024

	#create nodule mask
	nodule_mask = draw_circles(lung_img,cands,origin,new_spacing)

	for cand in cands:
		coord_x = float(cand[1])
		coord_y = float(cand[2])
		coord_z = float(cand[3])
		radius = int(np.ceil(float(cand[4])/2))

		world_coord = np.array((coord_z,coord_y,coord_x))
		#determine voxel coordinate given the worldcoordinate
		image_coord = world_2_voxel(world_coord,origin,new_spacing)
		print world_coord, image_coord
		ims_vis([normalizePlanes(lung_img[int(np.round(image_coord[0])),:,:]), nodule_mask[int(np.round(image_coord[0])),:,:]])
		ims_vis([normalizePlanes(lung_img[int(np.round(image_coord[0])), int(np.round(image_coord[1] - radius - 10)):int(np.round(image_coord[1] + radius + 10)), int(np.round(image_coord[2] - radius - 10)):int(np.round(image_coord[2] + radius + 10))]), \
		normalizePlanes(lung_img[int(np.round(image_coord[0] + radius + 10)), int(np.round(image_coord[1] - radius - 10)):int(np.round(image_coord[1] + radius + 10)), int(np.round(image_coord[2] - radius - 10)):int(np.round(image_coord[2] + radius + 10))])])
	# lung_img_512 = np.zeros((lung_img.shape[0], 512, 512))
	# # lung_mask_512 = np.zeros((lung_mask.shape[0], 512, 512))
	# nodule_mask_512 = np.zeros((nodule_mask.shape[0], 512, 512))

	# original_shape = lung_img.shape
	# for z in range(lung_img.shape[0]):
	# 	offset = (512 - original_shape[1])
	# 	upper_offset = np.round(offset/2)
	# 	lower_offset = offset - upper_offset

	# 	new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

	# 	lung_img_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img[z,:,:]
	# 	# lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:]
	# 	nodule_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]

    # save images.    
	# np.save(imageName + '_lung_img.npz', lung_img_512)
	# np.save(imageName + '_lung_mask.npz', lung_mask_512)
	# np.save(imageName + '_nodule_mask.npz', nodule_mask_512)

create_nodule_mask(img_path, cands_test)