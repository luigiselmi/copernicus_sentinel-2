#! /usr/bin/python
###############################################
# bigearthlib v.0.1.1 (29.08.2025)
###############################################
import numpy as np
import os
import sys
from osgeo import gdal, osr, ogr
import pathlib
import time
import rasterio
from rasterio.plot import show_hist
from rasterio.plot import show
import PIL
from PIL import Image, ImageDraw
from skimage import io
from skimage import exposure
from skimage.io import imread
import tifffile as tiff
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from matplotlib import cm, ticker
from matplotlib.colors import ListedColormap, LinearSegmentedColormap 
import zipfile
from zipfile import ZipFile
import warnings
warnings.filterwarnings('ignore')

'''
This script implements some functions to create PNG RGB images 
from TIFF images and masks of the BigEarthNet dataset.
'''
## ---------------------------------------------- Start of functions definition -----------------------------------------------
# The script is divided into four sections: 
# 1. Data collection
# 2. TIFF to PNG transformation
# 3. Compression
# 4. Normalization
# 5. Visualization
# 6. Statistics
#------------------------- 1) Data collection --------------------------------------------------
def read_band_name(band_name):
    '''
    Returns the information encoded in a TIFF image band file name:
    tile, patch, band, and date of acquisition.
    '''
    band = band_name[-7:-4]
    patch = band_name[-13:-8]
    tile = band_name[-25:-14]
    date = band_name[-47:-39]
    return tile, patch, band, date

def read_mask_name(mask_name):
    '''
    Returns the information encoded in a TIFF mask file name:
    tile, patch, band, and date of acquisition.
    '''
    patch = mask_name[-23:-18]
    tile = mask_name[-35:-24]
    date = mask_name[-57:-49]
    return tile, patch, date

def create_png_file_name(tile, patch, date):
    return tile + '_' + patch + '_' + date + '.png'

def create_mask_png_file_name(tile, patch, date):
    return tile + '_' + patch + '_' + date + '_mask.png'

def list_image_files(root_path, start_tile_index, end_tile_index):
    '''
    This function creates a list of tiles each containing
    lists of patches with three RGB bands each or a mask.
    The 2nd argument is the index of the first tile to be included. 
    The 3rd argument is the number of tiles to be returned. 
    '''
    tiles_list = []
    tiles_paths = [pathlib.Path(x) for x in root_path.iterdir() if x.is_dir()]
    for tile_path in tiles_paths[start_tile_index:end_tile_index]:
        # print(tile_path.name)
        patches_list = []
        for patches_path in tile_path.iterdir():
            bands_list = []
            for band_path in patches_path.iterdir():
                band_type = band_path.name[-7:]
                if (band_type == 'B02.tif' or band_type == 'B03.tif' or band_type == 'B04.tif'):
                    bands_list.append(band_path)
            patches_list.append(bands_list)
        tiles_list.append(patches_list)
    return tiles_list

def list_mask_files(root_path, start_tile_index, end_tile_index):
    '''
    This function, like the one for bands, creates a list of tiles 
    each containing lists of patches with a mask.
    The 2nd argument is the index of the first tile to be included. 
    The 3rd argument is the number of tiles to be returned. 
    '''
    tiles_list = []
    tiles_paths = [pathlib.Path(x) for x in root_path.iterdir() if x.is_dir()]
    #print('Number of tiles: ', len(tiles_paths))
    for tile_path in tiles_paths[start_tile_index:end_tile_index]:
        #print('Tile:', tile_path.name)
        patches_list = []
        for patch_path in tile_path.iterdir():
            mask_list = []
            for mask_path in patch_path.iterdir():
                file_type = mask_path.name[-7:]
                #print('File type: {}'.format(file_type))
                if (file_type == 'map.tif'):
                    mask_list.append(mask_path)
            patches_list.append(mask_list)
        tiles_list.append(patches_list)
    return tiles_list

def print_images_list(tiles_list): 
    '''
    Prints the content of the nested folders
    within the list passed as argument.
    tiles[patches[bands[]]]
    '''
    count = 0
    for patches_list in tiles_list:
        for bands_list in patches_list:
            for band in reversed(bands_list):
                print(band.name)
                count += 1
            print('\n')
        print('\n')
    return count
        
def print_masks_list(tiles_list): 
    '''
    Prints the content of the nested folders
    within the list passed as argument.
    tiles[patches[mask]]
    '''
    count = 0
    for patches_list in tiles_list:
        for mask_list in patches_list:
            for mask in mask_list:
                print(mask.name)
                count += 1
            print('\n')
        print('\n')
    return count

def get_raster_attributes(img_path):
    width = 0.0
    height = 0.0
    d_type = None
    crs = None
    transform = None
    with rasterio.open(img_path) as dataset:
        d_types = dataset.dtypes[0]
        print('dtypes: {}'.format(d_types))
        print('Number of bands: {:d}'.format(dataset.count))
        width = dataset.width
        height = dataset.height
        print('Band width: {:d}, band height: {:d}'.format(width, height))
        transform = dataset.transform
        print('Dataset affine transform:\n {}'.format(transform))
        crs = dataset.crs
        print('EPSG Coordinates Reference System: {}'.format(crs))
        bb_left = dataset.bounds.left
        bb_bottom = dataset.bounds.bottom
        bb_right = dataset.bounds.right
        bb_top = dataset.bounds.top
        print('Bounding box \n left: {:.2f}, \n bottom: {:.2f}, \n right: {:.2f}, \n top: {:.2f}'.format(bb_left, bb_bottom, bb_right, bb_top))                                                   
    return width, height, d_type, transform

#----------------------------2) TIFF to PNG transformation ----------------------------------

def normalize_uint8(data_array):
    '''
    This function transforms the bit depth of the TIFF bands from 16 to 8 (0-255).
    '''
    img_size = data_array.shape
    norm_array = (data_array - np.min(data_array)) * (255 / (np.max(data_array) - np.min(data_array)))
    norm_array_uint8 = norm_array.astype(np.uint8)
    return norm_array_uint8
    
def createPNG(source_path_list, target_path):
    '''
    This function creates a multiband PNG file from a list of GeoTIFF files 
    containing one band each. For an RGB file the source list shall contain three bands
    in the RGB order. For Sentinel-2 it is B04, B03, B02. The bin depth of each band is
    reduced from 16 bits to 8 bits. If the target file already exists
    it doesn't create a new one and will return 1, otherwise it will create a new raster
    and will return 0 
    '''
    SUCCESS = 0
    FAILURE = 1
    if (os.path.isfile(target_path)):
        return FAILURE 
        
    band_list = []
    with rasterio.open(source_path_list[0]) as source_dataset:
        width = source_dataset.width
        height = source_dataset.height
        count = len(source_path_list)

    #print('createPNG source_path_list', source_path_list)
    for raster_path in source_path_list:
        dataset = rasterio.open(raster_path)
        band = normalize_uint8(dataset.read(1))
        band_list.append(band)
        
        
    with rasterio.open(target_path,
                    mode='w',
                    driver='PNG',
                    height=height,
                    width=width,
                    count=count,
                    dtype='uint8') as target_dataset:
        band_index = 1
        for band in band_list:
            target_dataset.write(band, band_index)
            band_index += 1
    
    return SUCCESS

def createPNGfromArray(band_array_list, target_path):
    '''
    This function creates a multiband PNG file from a list of GeoTIFF files 
    containing one band each. For an RGB file the source list shall contain three bands
    in the RGB order. For Sentinel-2 it is B04, B03, B02. The bin depth of each band is
    reduced from 16 bits to 8 bits. If the target file already exists
    it doesn't create a new one and will return 1, otherwise it will create a new raster
    and will return 0 
    '''
    SUCCESS = 0
    FAILURE = 1
    if (os.path.isfile(target_path)):
        return FAILURE 
        
    width = band_array_list[0].shape[0]
    height = band_array_list[0].shape[1]
    count = len(band_array_list)    
        
    with rasterio.open(target_path,
                    mode='w',
                    driver='PNG',
                    height=height,
                    width=width,
                    count=count,
                    dtype='uint8') as target_dataset:
        band_index = 1
        for band in band_array_list:
            norm_band = normalize_uint8(band)
            target_dataset.write(norm_band, band_index)
            band_index += 1
    
    return SUCCESS

def createMaskPNG(source_path, target_path):
    '''
    This function creates a one band PNG file from a GeoTIFF mask file. 
    If the target file already exists it doesn't create a new one and 
    will return 1, otherwise it will create a new raster and will return 0. 
    We use dtype uint16 because mask pixel values can be > 255.
    '''
    #print('createMaskPNG source_path=', source_path)
    SUCCESS = 0
    FAILURE = 1
    if (os.path.isfile(target_path)):
        return FAILURE 
        
    with rasterio.open(source_path) as source_dataset:
        width = source_dataset.width
        height = source_dataset.height
        band = source_dataset.read(1)
        
    with rasterio.open(target_path,
                    mode='w',
                    driver='PNG',
                    height=height,
                    width=width,
                    count=1,
                    dtype='uint16') as target_dataset:
        band_index = 1
        target_dataset.write(band, band_index)

    return SUCCESS

def createPNGs(tiles_list):
    '''
    This function creates a PNG for each patch of the tiles
    in the list. The path of the PNG files is added to a list 
    that will be returned. In case a PNG file already exist it 
    only adds its path to the list.
    '''
    png_patches = []
    num_tiles = 0
    for patches_list in tiles_list:
        for bands_list in patches_list: 
            band_name = bands_list[0].name
            #print('Band name: {}'.format(band_name))
            tile, patch, band, date = read_band_name(band_name)
            patch_dir = bands_list[0].parent
            png_file_name = str(patch_dir) +  '/' + create_png_file_name(tile, patch, date)
            if (createPNG(bands_list, png_file_name) == 1):
                #print('The image PNG file already exists.')
                png_patches.append(png_file_name)
            else:
                png_patches.append(png_file_name)
        num_tiles += 1
        print('Tile image {:d} completed'.format(num_tiles))
    return png_patches

def createMaskPNGs(tiles_list):
    '''
    This function creates a PNG mask for each patch of the tiles
    in the list. The path of the PNG files is added to a list 
    that will be returned. In case a PNG file already exist it 
    only adds its path to the list.
    '''
    png_patches = []
    num_tiles = 0
    for patches_list in tiles_list:
        for patch_path in patches_list: 
            #print('Patch path: {}'.format(patch_path))
            patch_dir = patch_path[0].parent
            patch_name = patch_path[0].name
            #print('Patch name: {}'.format(patch_name))
            tile, patch, date = read_mask_name(patch_name)
            #print('Tile: {}, Patch: {}, Date: {}'.format(tile, patch, date))
            png_file_name = str(patch_dir) +  '/' + create_mask_png_file_name(tile, patch, date)
            #print('Mask file name: {}'.format(png_file_name))
            tiff_path_name = str(patch_path[0])
            #print('Tiff patch path name', tiff_path_name)
            if (createMaskPNG(tiff_path_name, png_file_name) == 1):
                #print('The mask PNG file already exists.')
                png_patches.append(png_file_name)
            else:
                png_patches.append(png_file_name)
        num_tiles += 1
        print('Tile mask {:d} completed'.format(num_tiles))
    return png_patches

def delete_files(file_list):
    '''
    Removes all the files in the list
    '''
    for file in file_list:
        file_path = pathlib.Path(file)
        file_path.unlink()  

def count_unique_occurrence(array):
    '''
    Computes the number of elements in an array that 
    have one of the unique values. Returns the list
    of unique values and the number of occurrences.
    This function can be replaced by the NumPy function
    np.unique(array, return_counts=True)
    '''
    occurrences = []
    unique_values = np.unique(array)
    for u in unique_values:
        occurrence = np.count_nonzero(array == u)
        occurrences.append(occurrence)
    return unique_values, occurrences

def resize_png(file_path, png_size):
    '''
    Transforms a PNG file into a tensor
    and returns a new tensor resized
    according to the png_size, e.g. (128, 128).
    '''
    img = tf_io.read_file(file_path)
    num_channnels = img.shape
    decoded_img = tf_io.decode_png(img, channels=num_channnels)
    resized_img = tf_image.resize(decoded_img, png_size)
    return resized_img

def mapCorineL3(source_path, target_path):
    '''
    This function creates a new target mask PNG file from a source mask file
    mapping the Corine2018 Level 3 color codes of the source to their index 
    in [1, 45]. If the target file already exists it doesn't create a new one 
    and will return 1, otherwise it will create a new raster and will return 0. 
    The dtype of the target file is uint8.
    '''
    #print('createMaskPNG source_path=', source_path)
    SUCCESS = 0
    FAILURE = 1
    if (os.path.isfile(target_path)):
        return FAILURE 
        
    with rasterio.open(source_path) as source_dataset:
        width = source_dataset.width
        height = source_dataset.height
        band = source_dataset.read(1)

    band = corine_l3_mask(band)
    
    with rasterio.open(target_path,
                    mode='w',
                    driver='PNG',
                    height=height,
                    width=width,
                    count=1,
                    dtype='uint8') as target_dataset:
        band_index = 1
        target_dataset.write(band, band_index)

    return SUCCESS

def mapCorineL3_list(source_folder, target_folder):
    '''
    This function creates new mask PNG files in the target folder from 
    source mask files in the source folder by mapping the Corine2018 
    level 3 color codes of the source files to their index in [1, 45]
    '''
    target_masks = []
    source_masks_folder_path = pathlib.Path(source_folder)
    source_masks = [str(file) for file in source_masks_folder_path.iterdir()]
    num_source_masks = len(source_masks)
    for source_mask in source_masks:
        #print('Source mask: {}', source_mask)
        target_mask_name = pathlib.Path(source_mask).name[:-4] + '_nc.png'    
        target_mask = target_folder + target_mask_name
        #print('Target mask: {}', target_mask)
        mapCorineL3(source_mask, target_mask)
        target_masks.append(target_mask)
    return target_masks

def mapCorineL1(source_path, target_path):
    '''
    This function creates a new target mask PNG file from a source mask file
    mapping the Corine2018 Level 3 color codes of the source to level 1. If 
    the target file already exists it doesn't create a new one and will return 
    1, otherwise it will create a new raster and will return 0. The dtype of 
    the target file is uint8.
    '''
    #print('createMaskPNG source_path=', source_path)
    SUCCESS = 0
    FAILURE = 1
    if (os.path.isfile(target_path)):
        return FAILURE 
        
    with rasterio.open(source_path) as source_dataset:
        width = source_dataset.width
        height = source_dataset.height
        band = source_dataset.read(1)

    band = corine_l1_mask(band)
    
    with rasterio.open(target_path,
                    mode='w',
                    driver='PNG',
                    height=height,
                    width=width,
                    count=1,
                    dtype='uint8') as target_dataset:
        band_index = 1
        target_dataset.write(band, band_index)

    return SUCCESS

def mapCorineL1_list(source_folder, target_folder):
    '''
    This function creates new mask PNG files in the target folder from 
    source mask files in the source folder by mapping the Corine2018 
    level 3 color codes of the source files to level 1
    '''
    target_masks = []
    source_masks_folder_path = pathlib.Path(source_folder)
    source_masks = [str(file) for file in source_masks_folder_path.iterdir()]
    num_source_masks = len(source_masks)
    #print('Num. source masks: ', num_source_masks)
    for source_mask in source_masks:
        #print('Source mask: ', source_mask)
        target_mask_name = pathlib.Path(source_mask).name[:-8] + 'l1_mask.png'    
        target_mask = target_folder + target_mask_name
        #print('Target mask: ', target_mask)
        mapCorineL1(source_mask, target_mask)
        target_masks.append(target_mask)
    return target_masks


def corine2018_l1_class_bucket(clc_code):
    '''
    This function returns the index of a bucket from 1 to 6
    to be used in place of a Corine2018 Level 3 class code. 
    The last code 999 does not belong to Corine2018 and is 
    used for pixels that were not classified. This function 
    works as the inverse of corine2018_l1_class_code(). If 
    the clc_code is not valid the function returns no values.
    '''
    corine2018_class_code = [
        [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142],
        [211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244],
        [311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335],
        [411, 412, 421, 422, 423],
        [511, 512, 521, 522, 523], 
        [999]]
    
    corine2018_l1_class_buckets = np.arange(1, 7, dtype='uint8')

    for c in range(1, 7):
        if (clc_code in corine2018_class_code[c - 1]):
            return c

def corine2018_l1_labels():
    '''
    This function simply returns the list of the 5+1 
    Corine2018 level 1 land cover labels plus one additional class,
    'Unclassified', used for pixels that were not classified.
    '''
    corine2018_level1_labels = [
        'Artificial surfaces',
        'Agricultural areas',
        'Forest and semi-natural areas',
        'Wetlands',
        'Water bodies',
        'Unclassified']
    
    return corine2018_level1_labels

def corine_l1_color_map():
    '''
    This function returns the 6 Corine2018 Level 1
    RGB color codes in hexadecimal format used for 
    the visualization of the land cover, plus an
    additional 255-255-255 (white) code to annotate
    pixels that are not classified.
    '''

    corine2018_l1_rgb_color_codes = [
        '230-000-077', '255-255-168', '128-255-000',
        '166-166-255', '000-204-242', '255-255-255']

    hex_color_map = []
    for color in corine2018_l1_rgb_color_codes:
        r = int(color[:3])
        g = int(color[4:7])
        b = int(color[8:11])
        hex = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        hex_color_map.append(hex)
    
    return hex_color_map

#-------------------------------------3) Compression ----------------------------------------------------

def zip_pngs(pngs_list, target_zip_file):
    '''
    This function can be used to compress all the PNG files
    created using the createPNGs function into a zip file
    '''
    with ZipFile(target_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipObj:
        for png in pngs_list:
            png_path = pathlib.Path(png)
            zipObj.write(png, arcname=png_path.name)

def unzip_pngs(source_zip_file, target_folder):
    with ZipFile(source_zip_file, 'r') as zipObj:
        zipObj.extractall(path=f'{target_folder}')

## ------------------------------------------ 4) Normalization ------------------------------------
def get_image_array(img_path):
    '''
    This function returns a NumPy array
    of the image. For one band it returns 
    a 2D array with shape (height, width).
    For a RGB image with three bands it 
    returns a 3D array with shape (channels, height, width) 
    '''
    with gdal.Open(img_path) as image_ds:
        image_array = image_ds.ReadAsArray()
    return image_array
    
def norm_image(image_array):
    '''
    This function takes a NumPy array as input, computes min and max
    and returns a normalized array with values in [0, 1]
    '''
    max_image = image_array.max()
    min_image = image_array.min()
    image_array_norm = (image_array - min_image) / (max_image - min_image + 1)
    return image_array_norm
    
## --------------------------------------------- 5) Visualization ---------------------------------------

def plot_examples(images_list, masks_list, start=0, end=10):
    '''
    Plots images and masks in two columns.
    '''
    row_start = start
    row_end = end
    num_rows = row_end - row_start
    fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(10, 10), layout='tight')
    corine2018_l3_color_map = ListedColormap(corine_l3_color_map())
    for i in range(num_rows):
        img = Image.open(images_list[row_start + i])
        img_name = pathlib.Path(images_list[row_start + i]).name[:-4]
        msk = Image.open(masks_list[row_start + i])
        msk_name = pathlib.Path(masks_list[row_start + i]).name[:-9]
        axs[i, 0].set_axis_off()
        axs[i, 1].set_axis_off()
        axs[i, 0].imshow(img)
        axs[i, 0].set_title(img_name)
        axs[i, 1].imshow(msk, cmap=corine2018_l3_color_map)
        axs[i, 1].set_title(msk_name)

def corine_l3_color_map():
    '''
    This function returns the 44 Corine2018 RGB
    color codes in hexadecimal format used for 
    the visualization of the land cover, plus an
    additional 255-255-255 (white) code to annotate
    pixels that are not classified.
    '''

    corine2018_rgb_color_codes = [
        '230-000-077', '255-000-000', '204-077-242', '204-000-000', '230-204-204', '230-204-230', '166-000-204', '166-077-000', '255-077-255', '255-166-255', '255-230-255', 
        '255-255-168', '255-255-000', '230-230-000', '230-128-000', '242-166-077', '230-166-000', '230-230-077', '255-230-166', '255-230-077', '230-204-077', '242-204-166', 
        '128-255-000', '000-166-000', '077-255-000', '204-242-077', '166-255-128', '166-230-077', '166-242-000', '230-230-230', '204-204-204', '204-255-204', '000-000-000', '166-230-204', 
        '166-166-255', '077-077-255', '204-204-255', '230-230-255', '166-166-230', 
        '000-204-242', '128-242-230', '000-255-166', '166-255-230', '230-242-255', 
        '255-255-255']

    hex_color_map = []
    for color in corine2018_rgb_color_codes:
        r = int(color[:3])
        g = int(color[4:7])
        b = int(color[8:11])
        hex = "#{:02x}{:02x}{:02x}".format(r,g,b)
        hex_color_map.append(hex)
    
    return hex_color_map

def corine2018_l3_labels():
    '''
    This function simply returns the list of the 44 
    Corine2018 land cover labels plus one additional class,
    'Unclassified', used for pixels that were not classified.
    '''
    corine2018_level3_labels = [
        'Continuous urban fabric',
        'Discontinuous urban fabric',
        'Industrial or commercial units',
        'Road and rail networks and associated land',
        'Port areas',
        'Airports',
        'Mineral extraction sites',
        'Dump sites',
        'Construction sites',
        'Green urban areas',
        'Sport and leisure facilities',
        'Non-irrigated arable land',
        'Permanently irrigated land',
        'Rice fields',
        'Vineyards',
        'Fruit trees and berry plantations',
        'Olive groves',
        'Pastures',
        'Annual crops associated with permanent crops',
        'Complex cultivation patterns',
        'Land principally occupied by agriculture, with significant areas of natural vegetation',
        'Agro-forestry areas',
        'Broad-leaved forest',
        'Coniferous forest',
        'Mixed forest',
        'Natural grasslands',
        'Moors and heathland',
        'Sclerophyllous vegetation',
        'Transitional woodland-shrub',
        'Beaches, dunes, sands',
        'Bare rocks',
        'Sparsely vegetated areas',
        'Burnt areas',
        'Glaciers and perpetual snow',
        'Inland marshes',
        'Peat bogs',
        'Salt marshes',
        'Salines',
        'Intertidal flats',
        'Water courses',
        'Water bodies',
        'Coastal lagoons',
        'Estuaries',
        'Sea and ocean',
        'Unclassified']
    return corine2018_level3_labels

def corine_l3_mask(mask_array):
    '''
    This function maps the Corine2018 values of a mask
    to a set indexes, from 1 to 45, that corresponds to the
    Corine2018 classes.
    '''
    size = mask_array.shape
    unif_mask = np.zeros(size, dtype=np.int8)
    unique_values = np.unique(mask_array) 
    for u in unique_values:
        t_mask = (mask_array == u)
        unif_mask[t_mask] = corine2018_l3_class_bucket(u)
    return unif_mask

def corine_l1_mask(mask_array):
    '''
    This function maps the Corine2018 L3 values of a mask
    to Level 1 (5 + 1 classes and color codes).
    '''
    size = mask_array.shape
    unif_mask = np.zeros(size, dtype=np.int8)
    unique_values = np.unique(mask_array) 
    for u in unique_values:
        t_mask = (mask_array == u)
        unif_mask[t_mask] = corine2018_l1_class_bucket(u)
    return unif_mask


## ---------------------------------------------- 6) Statistics
def corine2018_l3_class_code(index):
    '''
    This function returns the Corine2018 Land Cover classification 
    code at the 3rd level given its index (from 1 to 45). The last code
    999 does not belong to Corine2018 and is used for pixels that were
    not classified.
    '''
    SUCCESS = 0
    FAILURE = 1
    if (index < 1 or index > 45):
        print('The index must be a number between 1 and 44')
        return FAILURE
    
    corine2018_class_code = [
        111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142,
        211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244,
        311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335,
        411, 412, 421, 422, 423,
        511, 512, 521, 522, 523, 
        999]
    
    return corine2018_class_code[index - 1]
    
def corine2018_l3_class_bucket(clc_code):
    '''
    This function return the index of a bucket from 1 to 45
    to be used in place of a Corine2018 class code. The last code
    999 does not belong to Corine2018 and is used for pixels that 
    were not classified. This function works as the inverse of 
    corine2018_l3_class_code().
    '''
    corine2018_class_code = [
        111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142,
        211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244,
        311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335,
        411, 412, 421, 422, 423,
        511, 512, 521, 522, 523, 
        999]
    
    clc_code_index = corine2018_class_code.index(clc_code)
    corine2018_class_buckets = np.arange(1,46, dtype='uint8')
    bucket = corine2018_class_buckets[clc_code_index]
    return bucket

def collect_statistics(root_path, start_tile_index, end_tile_index, print_msg=False):
    '''
    This function collects the unique values in each BigEarthNetv2 mask file
    and put them in a bucket list from 1 to 44 in order to count how many masks
    contain each one of the Corine2018 classes. The function return an array of
    the 44 buckets with the number of masks for each class.
    '''
    num_mask_files = 0
    corine2018_buckets = np.zeros(45)
    tiles_paths = [pathlib.Path(x) for x in root_path.iterdir() if x.is_dir()]
    if (print_msg):
        print('Number of tiles: ', len(tiles_paths))
    for tile_path in tiles_paths[start_tile_index:end_tile_index]:
        if (print_msg):
            print('Tile:', tile_path.name)
        patches_paths = [pathlib.Path(x) for x in tile_path.iterdir() if x.is_dir()]
        if (print_msg):
            print('Number of patches per tile: ', len(patches_paths))
        for patch_path in patches_paths:
            if (print_msg):
                print('Patch: ', patch_path.name)
            for mask_path in patch_path.iterdir():
                file_type = mask_path.name[-7:]
                if (file_type == 'map.tif'):
                    mask_png = rasterio.open(mask_path)
                    mask_array = mask_png.read(1)
                    unique_values = np.unique(mask_array)
                    num_mask_files += 1
                    if (print_msg):
                        print('Unique values: ', unique_values)
                    for u in unique_values:
                        bucket = corine2018_class_bucket(u)
                        bucket_value = corine2018_buckets[bucket - 1]
                        corine2018_buckets[bucket - 1] = bucket_value + 1
    return corine2018_buckets

def save_statistics(bucket_array, file_path):
    '''
    Saves the bucket array in a txt file,
    one value per line.
    '''
    data = bucket_array.tolist()
    with open(file_path, 'w') as f:
        for element in data:
            f.write(f'{element}\n')

def read_statistics(file_path):
    '''
    This function reads the statistics file: a txt
    file with one value per line. Each value represents
    the number of images that contains pixels of a class
    in the Corine2018 classification code.
    '''
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        bucket_array = np.zeros(len(lines))
        index = 0
        for line in lines:
            bucket_array[index] = line.strip()
            index += 1
        return bucket_array
## ---------------------------------------------- End of functions definition -----------------------------------------------
