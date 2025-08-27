#! /usr/bin/python
###############################################
# satlib v.0.1 (27.08.2025)
###############################################
import os
import numpy as np
from osgeo import gdal, osr, ogr
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import sys
#from ipyleaflet import Map, Marker, basemaps, basemap_to_tiles, Rectangle, DrawControl, LayersControl, GeoData
#from ipywidgets import Layout
import json
from os import listdir
from os.path import isfile, join
from PIL import Image
import shutil

# Constants
COPERNICUS_VHR2015_IMAGERY_ROOT_FOLDER = '/eos/jeodpp/data/SRS/Copernicus/DAP/DWH-2/VHR_IMAGE_2015/source/'
EOS_TRAINING_DATA_FOLDER = '/eos/jeodpp/data/projects/3D-BIG/data/processed_data/rooftop-type-trainingDataset/dbsm-rti'
TRAINING_FOLDER = EOS_TRAINING_DATA_FOLDER + '/train/'
BASE_FOLDER = EOS_TRAINING_DATA_FOLDER + '/base/'
IMAGERY_FOLDER_PATH = '/scratch/luselmi/dbsm-rti/'
VHR2015_INDEX_FILE_PATH = IMAGERY_FOLDER_PATH + 'vhr2015/vhr2015_tileindex_all.gpkg'
OSM_DATA_FOLDER = osm_data_path = IMAGERY_FOLDER_PATH + 'openstreetmap/'
SUBSETS_FOLDER = 'subsets/'
PATCHES_FOLDER = 'patches/'
FOOTPRINTS_FOLDER = 'footprints/'
SUBSETS_GDF_FILE_PATH = IMAGERY_FOLDER_PATH + 'subsets.geojson'

def create_bbox_vertices(nw_vertex, se_vertex, crs=4326):
    """
    Transforms two pairs of points, NW and SE, in the WGS84 crf given in the form 
    Point(lon, lat), into a GeoPandas data frame containing 
    the two input points plus the NE and SW points 
    """
    vertex_gdf = None
    if (crs == 4326):
        nw_vertex = project_epsg3035(nw_vertex.x, nw_vertex.y)
        se_vertex = project_epsg3035(se_vertex.x, se_vertex.y)
    ne_vertex = Point(se_vertex.x, nw_vertex.y)
    sw_vertex = Point(nw_vertex.x, se_vertex.y)
    vertex_names = ['north west', 'south east', 'north east', 'south west']
    vertex_geometries = [nw_vertex, se_vertex, ne_vertex, sw_vertex]
    vertex_dict = {'name': vertex_names, 'geometry': vertex_geometries}
    vertex_gdf = gpd.GeoDataFrame(vertex_dict, crs=3035)
    
    return vertex_gdf

def create_bbox_polygon(nw_vertex, se_vertex, crs=4326):
    """
    Transforms two pairs of points, NW and SE, in the WGS84 or 
    EPSG:3035 crs, given in the form Point(lon, lat) or Point(x, y), 
    into a GeoPandas data frame containing the polygon of the bounding 
    box defined by the two vertices.
    """
    bbox_gdf = None
    
    if (crs == 4326):
        nw_vertex = project_epsg3035(nw_vertex.x, nw_vertex.y)
        se_vertex = project_epsg3035(se_vertex.x, se_vertex.y)
    ne_vertex = Point(se_vertex.x, nw_vertex.y)
    sw_vertex = Point(nw_vertex.x, se_vertex.y)
    bbox_geometries = [nw_vertex, sw_vertex, se_vertex, ne_vertex, nw_vertex]
    bbox_polygon = Polygon(bbox_geometries)
    bbox_dict = {'name': 'Area of Interest', 'geometry': bbox_polygon}
    bbox_gdf = gpd.GeoDataFrame(bbox_dict, crs=3035, index=range(0,1))
        
    return bbox_gdf

def project_epsg3035(lon, lat):
    """
    This function transforms a point coordinates (lat, lon) from WGS84 (EPSG:4326) 
    to EPSG:3035 (x, y)
    """
    
    point_wkt = 'POINT (' + str(lat) + ' ' + str(lon) + ')'
    point = ogr.CreateGeometryFromWkt(point_wkt)
    source_ref = osr.SpatialReference()
    source_ref.ImportFromEPSG(4326)
    target_ref = osr.SpatialReference()
    target_ref.ImportFromEPSG(3035)
    transform_ref = osr.CoordinateTransformation(source_ref, target_ref)
    point.Transform(transform_ref)
    point3035 = point.GetPoint()
    x = point3035[1]
    y = point3035[0]
    return Point(x, y)

def image_date_tile(file_rel_path):    
    """
    This function extracts four fields from the file relative path:
    1) country code
    2) sub region
    3) image provider
    4) acquisition day (as YYYYMMDD)
    5) scene's tile index  
    """
    path_elements = file_rel_path.split('/')
    country_code = path_elements[0]
    sub_region = path_elements[1]
    provider = path_elements[3][:2]
    file_name = path_elements[-1:].pop()
    file_name_parts = file_name.split('_')
    acquisition_day = None
    tile_index = None
    if (provider == 'EW'): # WorldView Imagery
        acquisition_datetime = file_name_parts[3][:8]
        acquisition_day = acquisition_datetime[:8]
        tile_index = file_name_parts[1][:4]
    
    if (provider == 'PH'):
        acquisition_day = file_name_parts[3][:8]
        tile_index = file_name_parts[6][-8:-4]
        
    return country_code, sub_region, provider, acquisition_day, tile_index

def ms_image_path(ms_imagery_gdf, pan_day, pan_tile):
    """
    This function returns the absolute path of a multispectral image
    that has been acquired the same day of a panchromatic image 
    and has the same tile index
    """
    ms_path_list = []
    for i in range(0, len(ms_imagery_gdf)):
        ms_index = ms_imagery_gdf.index[i]
        ms_rel_path = ms_imagery_gdf.loc[ms_index]['relpath']
        country_code, sub_region, provider, acquisition_day, tile = image_date_tile(ms_rel_path)
        if (acquisition_day == pan_day and tile == pan_tile):
            ms_path = ms_imagery_gdf.loc[ms_index]['location']
            ms_path_list.append(ms_path)
    return ms_path_list

def getRasterBands(ds, origin_col, origin_row, cols, rows):
    """
    This function extracts a patch from each
    band in an image starting from the patch origin
    rows and columns
    """
    num_ms_bands = ds.RasterCount
    ms_data = []
    for i in range(0, num_ms_bands):
        ms_data.append(np.empty((rows, cols), dtype=np.uint16)) 
        
    for i in range(0, num_ms_bands):
        ms_band = ds.GetRasterBand(i + 1)
        ms_band.ReadAsArray(origin_col,
                       origin_row, 
                       cols,
                       rows,
                       buf_obj=ms_data[i])
    
    return ms_data
    

def normalize(array):
    array_norm = (array - array.min()) / (array.max() - array.min())
    return array_norm

def subset_image_origin(source_ds, patch_index_x, patch_index_y, patch_length_x, patch_length_y):
    """
    A function to compute the coordinates of the origin 
    of the image patch given its indices and the lengths 
    of the image patch.
    """
    subset_origin_x, resolution_x, row_rotation, subset_origin_y, col_rotation, resolution_y = subset_ds.GetGeoTransform()
    patch_length_cols = round(patch_length_x / resolution_x)
    patch_length_rows = round(patch_length_y / abs(resolution_y))
    patch_origin_x = subset_origin_x + patch_index_x * patch_length_x
    patch_origin_y = subset_origin_y - patch_index_y * patch_length_y
    return patch_origin_x, patch_origin_y

def create_subset(source_ds, subset_origin_x, subset_origin_y, subset_length_x, subset_length_y, subset_path, num_bands=3):
    """
    This function returns a subset of a multispectral source image. The source image is passed through its data source.
    The other arguments are the number of bands to be returned in the subset image (default value 3), the coordinates of 
    the origin of the subset, in a projected reference system, given in the same units of the source image, the legth of 
    the subset area in the x and y directions, and the path where the subset image will be saved. 
    version 0.0.1
    """
    source_origin_x, resolution_x, row_rotation, source_origin_y, col_rotation, resolution_y = source_ds.GetGeoTransform()
    num_source_bands = source_ds.RasterCount
    #print('Subset path: {:s}'.format(subset_path))
    
    if (num_bands > num_source_bands) :
        print('Number of bands in source image lower than for the subset image')
        sys.exit(1)
    
    # indices (col and row) of the origin of the subset
    subset_rel_origin_x = subset_origin_x - source_origin_x 
    subset_rel_origin_y = source_origin_y - subset_origin_y
    subset_origin_col = int(subset_rel_origin_x / resolution_x)
    subset_origin_row = int(subset_rel_origin_y / resolution_x)
    #print('Origin\ncol: {}\nrow: {}'.format(subset_origin_col, subset_origin_row))

    # number of rows and columns of the subset
    subset_cols = round(subset_length_x / resolution_x)
    subset_rows = round(subset_length_y / abs(resolution_y))
    #print('subset cols: {:d}, subset rows: {:d}'.format(subset_cols, subset_rows))

    # coordinates of the subset origin
    subset_transform = [subset_origin_x, 
                      resolution_x, 
                      row_rotation, 
                      subset_origin_y, 
                      col_rotation,
                      resolution_y ]
    
    gtiff_driver = gdal.GetDriverByName('GTiff')
    subset_ds = gtiff_driver.Create(subset_path, subset_cols, subset_rows, num_bands, gdal.GDT_UInt16)
    subset_ds.SetProjection(source_ds.GetProjection())
    subset_ds.SetGeoTransform(subset_transform)
    
    subset_data = []
    for i in range(0, num_bands):
        subset_data.append(np.empty((subset_rows, subset_cols), dtype=np.uint16)) 
        
    for i in range(0, num_bands):
        source_band = source_ds.GetRasterBand(i + 1)
        source_band.ReadAsArray(subset_origin_col,
                       subset_origin_row, 
                       subset_cols,
                       subset_rows,
                       buf_obj=subset_data[i])
        
    for i in range(0, num_bands):
        subset_band = subset_ds.GetRasterBand(i + 1)
        subset_band.WriteArray(subset_data[i])
    
    subset_ds.FlushCache()
    subset_ds = None
    
def image_bbox(img_path, crs=3035):
    """
    This function returns the bounding box of an image
    as a GeoDataFrame
    """
    img_ds = gdal.Open(img_path)
    num_bands = img_ds.RasterCount
    img_cols = img_ds.RasterXSize
    img_rows = img_ds.RasterYSize
    img_origin_x, resolution_x, row_rotation, img_origin_y, col_rotation, resolution_y = img_ds.GetGeoTransform()
    resolution_x = round(resolution_x, 1)
    resolution_y = round(resolution_y, 1)
    img_length_x = img_cols * resolution_x
    img_length_y = img_rows * resolution_y
    img_se_x = img_origin_x + img_length_x  
    img_se_y = img_origin_y + img_length_y
    img_vertex_nw = Point(img_origin_x, img_origin_y)
    img_vertex_se = Point(img_se_x, img_se_y)
    img_bbox_gdf = create_bbox_polygon(img_vertex_nw, img_vertex_se, crs=crs)
    return img_bbox_gdf

def polygon_within(A_gdf, B_gdf):
    """
    This function returns a GeoDataFrame containing the polygons
    of A that lay within the polygon B
    """
    if (A_gdf.crs.name == 'WGS 84'):
        A_gdf.to_crs('epsg:3035', inplace=True)
        
    polygons_within_B_list = A_gdf.within(B_gdf.loc[0, 'geometry'])
    #if( not polygons_within_B_list.any()):
    #    return None
    
    A_gdf['within B'] = polygons_within_B_list
    polygons_within_B_gdf = A_gdf[A_gdf['within B'] == True]
    return polygons_within_B_gdf

def envelope(A_gdf):
    """
    This function return a set of polygons that are the envelope
    of the polygons in the input GeoDataFrame
    """
    geometries = []
    for polygon in A_gdf['geometry']:
        geometries.append(polygon.envelope)

    B_roof = A_gdf['roof:shape'].tolist()
    B_id = A_gdf['id'].tolist()
    
    B_dict = {'id': B_id, 'roof:shape': B_roof, 'geometry': geometries}
    B_gdf = gpd.GeoDataFrame(B_dict, crs='EPSG:3035')
    return B_gdf

def polygon_origin_size(pol_geometry):
    """
    This function returns the x, y coordinates of the origin
    of an envelope polygon and the length of the x and y 
    directions
    """
    minx, miny, maxx, maxy = pol_geometry.bounds
    nw_x = minx
    nw_y = maxy
    se_x = maxx
    se_y = miny
    nw_vertex = Point(nw_x, nw_y)
    se_vertex = Point(se_x, se_y)
    length_x = se_vertex.x - nw_vertex.x
    length_y = nw_vertex.y - se_vertex.y
    return nw_x, nw_y, length_x, length_y

def polygon_pixel_origin_size(pol_geometry, pixel_resolution):
    """
    This function returns the column and row of the origin
    of an envelope polygon and the number of columns and rows
    of the x and y directions
    """
    minx, miny, maxx, maxy = pol_geometry.bounds
    nw_x = minx
    nw_y = maxy
    se_x = maxx
    se_y = miny
    nw_vertex = Point(nw_x, nw_y)
    se_vertex = Point(se_x, se_y)
    length_x = se_vertex.x - nw_vertex.x
    length_y = nw_vertex.y - se_vertex.y
    nw_col = round(nw_vertex.x / pixel_resolution)
    nw_row = round(nw_vertex.y / pixel_resolution)
    cols = round(length_x / pixel_resolution)
    rows = round(length_y / pixel_resolution)
    return nw-col, nw_row, cols, rows

def polygon_nw_se(pol_geometry):
    """
    This function returns the NW and SE vertices
    of a polygon
    """
    minx, miny, maxx, maxy = pol_geometry.bounds
    nw_x = minx
    nw_y = maxy
    se_x = maxx
    se_y = miny
    vertex_nw = Point(nw_x, nw_y)
    vertex_se = Point(se_x, se_y)
    return vertex_nw, vertex_se

def padding_nw_se(env_nw_vertex, env_se_vertex, resolution_x = 0.5, resolution_y = -0.5):
    """
    This function returns the padded NW and SE vertices of
    those passed as arguments. The padding depends on the 
    standard size of a squared bounding box, e.g. 64x64 pixels
    """
    stand_img_size_rows = 64
    stand_img_size_cols = 64
    stand_img_length_x = stand_img_size_cols * resolution_x
    stand_img_length_y = stand_img_size_rows * abs(resolution_y)
    
    env_pad_rows_up = env_pad_rows_bottom = 0
    env_pad_cols_left = env_pad_cols_right = 0
    
    pad_nw_vertex_x = env_nw_vertex.x
    pad_nw_vertex_y = env_nw_vertex.y
    pad_se_vertex_x = env_se_vertex.x
    pad_se_vertex_y = env_se_vertex.y
    
    env_length_x = env_se_vertex.x - env_nw_vertex.x
    env_length_y = env_nw_vertex.y - env_se_vertex.y 
    env_cols = round(env_length_x / resolution_x)
    env_rows = round(env_length_y / abs(resolution_y))
    
    # case 1
    if ((env_cols <= stand_img_size_cols) & (env_rows <= stand_img_size_rows)):
        #print('env_cols: {}\nenv_rows: {}'.format(env_cols, env_rows))
        env_pad_cols_left = round((stand_img_size_cols - env_cols) / 2)
        env_pad_cols_right = stand_img_size_cols  - env_pad_cols_left - env_cols
        env_pad_rows_up = round((stand_img_size_rows - env_rows) / 2)
        env_pad_rows_bottom = stand_img_size_rows  - env_pad_rows_up - env_rows

    # case 2
    if ((env_rows > stand_img_size_rows) & (env_rows > env_cols)):
        env_pad_cols_left = round((env_rows - env_cols) / 2)
        env_pad_cols_right = env_rows  - env_pad_cols_left - env_cols    
        
    # case 3
    if ((env_cols > stand_img_size_cols) & (env_cols > env_rows)):
        env_pad_rows_up = round((env_cols - env_rows) / 2)
        env_pad_rows_bottom = env_cols  - env_pad_rows_up - env_rows    
    
    # new origin NW
    pad_vertex_nw_x = env_nw_vertex.x - env_pad_cols_left * resolution_x
    pad_vertex_nw_y = env_nw_vertex.y - env_pad_rows_up * resolution_y
    pad_vertex_nw = Point(pad_vertex_nw_x, pad_vertex_nw_y)
    
    # new vertex SE
    pad_vertex_se_x = env_se_vertex.x + env_pad_cols_left * resolution_x
    pad_vertex_se_y = env_se_vertex.y + env_pad_rows_up * resolution_y
    pad_vertex_se = Point(pad_vertex_se_x, pad_vertex_se_y)
    
    return pad_vertex_nw, pad_vertex_se

def create_patches(country_code,subregion, acquisition_day, tile, subset_id_str, patch_length_x=200, patch_length_y=200):
    """
    This function creates image patches from a subset image
    The default size of the patch is 200m x 200m 
    """
    pansharp_path = subset_file_path(country_code,subregion, acquisition_day, tile, subset_id_str, 'PAN_MODIFIED')
    patches_path = sub_folder_path(country_code,subregion, acquisition_day, tile, subset_id_str, image_content_type = 'PATCH')
    subset_ds = gdal.Open(pansharp_path)
    
    subset_cols = subset_ds.RasterXSize
    subset_rows = subset_ds.RasterYSize
    subset_origin_x, resolution_x, row_rotation, subset_origin_y, col_rotation, resolution_y = subset_ds.GetGeoTransform()
    
    subset_length_x = subset_cols * resolution_x
    subset_length_y = subset_rows * abs(resolution_y)
        
    patch_length_cols = round(patch_length_x / resolution_x)
    patch_length_rows = round(patch_length_y / abs(resolution_y))
    
    max_num_patches_x = int(subset_length_x / patch_length_x)
    max_num_patches_y = int(abs(subset_length_y) / patch_length_y)
    img_list = []
    for patch_index_x in range(0, max_num_patches_x):
        for patch_index_y in range(0, max_num_patches_y):
            patch_origin_x, patch_origin_y = subset_image_origin(subset_ds, patch_index_x, patch_index_y, patch_length_x, patch_length_y)
            patch_file_name = patch_file_path(country_code,subregion, acquisition_day, tile, subset_id_str, patch_index_x, patch_index_y)
            create_subset(subset_ds, patch_origin_x, patch_origin_y, patch_length_x, patch_length_y, patch_file_name)
            img_list.append(patch_file_name)
    
    subset_ds = None
    return img_list

def extract_building_images(patch_path, patch_index_str, osm_data, footprints_path):
    """
    This function takes as input an image data source and the path
    of a GeoJSON file with the building footprints in the same area
    and returns a list of images of each building
    """
    patch_ds = gdal.Open(patch_path)
    patch_bbox_gdf = image_bbox(patch_path)
    buildings_gdf = osm_data[['id', 'roof:shape', 'geometry']]
    buildings_within_patch_gdf = polygon_within(buildings_gdf, patch_bbox_gdf)
    img_list = []
    if (buildings_within_patch_gdf.empty == False):
        #num_buildings = buildings_within_patch_gdf.shape[0]
        #if (num_buildings > 0):
        env_buildings_gdf = envelope(buildings_within_patch_gdf)
        for footprint_index in env_buildings_gdf.index:
            env_geometry = env_buildings_gdf.iloc[footprint_index]['geometry']
            roof_shape = env_buildings_gdf.iloc[footprint_index]['roof:shape']
            vertex_nw, vertex_se = polygon_nw_se(env_geometry)
            env_polygon_gdf = create_bbox_polygon(vertex_nw, vertex_se, crs=3035)
            pad_vertex_nw, pad_vertex_se = padding_nw_se(vertex_nw, vertex_se)
            pad_env_length_x = pad_vertex_se.x - pad_vertex_nw.x
            pad_env_length_y = pad_vertex_nw.y - pad_vertex_se.y
            building_img_path = footprints_path + 'building_' + patch_index_str + '_' + str(footprint_index) + '_' + roof_shape + '.tif'
            img_list.append(building_img_path)
            create_subset(patch_ds, pad_vertex_nw.x, pad_vertex_nw.y, pad_env_length_x, pad_env_length_y, building_img_path)
    patch_ds = None
    return img_list

def map_data(center_map_lat, center_map_lon, osm_gdf_in, osm_bbox_gdf_in, bbox_gdf_in):
    """
    This function return a map to be used to show the geometries 
    inside a GeoPandas dataframe and to select an area of interest
    by drawing its bounding box. The data must use the EPSG4326 crs.
    Here follows a description of the input variables:
    - center_map_lat, center_map_lon : latitude and longitude of the center of the map where the data will be projected 
    - osm_gdf_in : dataset of the building footprints (polygons) inside the GeoJson file
    - osm_bbox_gdf_in : bounding box of the area of interest in the GeoJson file  
    - bbox_gdf_in : dataset of the subsets that are included in the area of interest 
    """
    m = Map(center=(center_map_lat, center_map_lon), zoom = 11, basemap= basemaps.OpenStreetMap.Mapnik)
    
    # OSM data
    if (osm_gdf_in.crs.name != 'WGS 84'):
        osm_gdf = osm_gdf_in.to_crs(4326, inplace=False)
    else:
        osm_gdf = osm_gdf_in
        
    osm_data = GeoData(geo_dataframe = osm_gdf,
                   style={'color': 'orange', 'fillColor': 'orange', 'opacity':1.0, 'weight':1.0, 'dashArray':'2', 'fillOpacity':1.0},
                   hover_style={'fillColor': 'blue' , 'fillOpacity': 1.0},
                   name = 'OSM data')
    m.add(osm_data)
    
    # OSM bounding box (area of interest)
    if (osm_bbox_gdf_in.crs.name != 'WGS 84'):
        osm_bbox_gdf = osm_bbox_gdf_in.to_crs(4326, inplace=False)
    else:
        osm_bbox_gdf = osm_bbox_gdf_in
    osm_bbox_data = GeoData(geo_dataframe = osm_bbox_gdf,
                   style={'color': 'green', 'fillColor': 'green', 'opacity':1.0, 'weight':1.0, 'dashArray':'2', 'fillOpacity':0.3},
                   hover_style={'fillColor': 'blue' , 'fillOpacity': 0.5},
                   name = 'OSM bbox')
    m.add(osm_bbox_data)
        
    # Subsets    
    if (bbox_gdf_in is not None): 
        if (bbox_gdf_in.crs.name != 'WGS 84'):
            bbox_gdf = bbox_gdf_in.to_crs(4326, inplace=False)
        else:
            bbox_gdf = bbox_gdf_in
        
    bbox_data = GeoData(geo_dataframe = bbox_gdf,
                   style={'color': 'green', 'fillColor': 'green', 'opacity':1.0, 'weight':1.0, 'dashArray':'2', 'fillOpacity':0.3},
                   hover_style={'fillColor': 'blue' , 'fillOpacity': 0.5},
                   name = 'bboxes')
    m.add(bbox_data)
    
    draw_control = DrawControl()
    draw_control.rectangle = {
       "shapeOptions": {
            "fillColor": "red",
            "color": "red",
            "fillOpacity": 0.3
        }
    }

    feature_collection = {
        'type': 'FeatureCollection',
        'features': []
    }

    def handle_draw(self, action, geo_json):
        """Do something with the GeoJSON when it's drawn on the map"""    
        feature_collection['features'].append(geo_json)

    draw_control.on_draw(handle_draw)

    m.add(draw_control)
    m.add(LayersControl())
    return m, feature_collection


def map_images(center_map_lat, center_map_lon, osm_gdf_in, pan_images_gdf_in, aoi_gdf_in):
    """
    This function return a map to be used to show the geometries of the panchromatic images 
    that overlap with the subset of an area of interest. The data must use the EPSG4326 crs
    that is WGS84. Here follows a description of the input variables:
    - center_map_lat, center_map_lon : latitude and longitude of the center of the map where the data will be projected 
    - osm_gdf_in : dataset of the building footprints (polygons) inside the GeoJson file
    - pan_images_gdf_in : bounding box of the panchromatic images that intersect a subset of the area of interest  
    - aoi_gdf_in : dataset of the subset that is included in the area of interest 
    """
    m = Map(center=(center_map_lat, center_map_lon), zoom = 11, basemap= basemaps.OpenStreetMap.Mapnik)
    
    # plot the OSM data
    if (osm_gdf_in.crs.name != 'WGS 84'):
        osm_gdf = osm_gdf_in.to_crs(4326, inplace=False)
    else:
        osm_gdf = osm_gdf_in
        
    osm_data = GeoData(geo_dataframe = osm_gdf,
                   style={'color': 'orange', 'fillColor': 'orange', 'opacity':1.0, 'weight':1.0, 'dashArray':'2', 'fillOpacity':1.0},
                   hover_style={'fillColor': 'blue' , 'fillOpacity': 1.0},
                   name = 'OSM data')
    m.add(osm_data)
    
    ## plot the bounding box of the panchromatic images
    if (pan_images_gdf_in is not None): 
        if (pan_images_gdf_in.crs.name != 'WGS 84'):
            pan_images_gdf = pan_images_gdf_in.to_crs(4326, inplace=False)
        else:
            pan_images_gdf = pan_images_gdf_in
            
    layers = create_gdf_list(pan_images_gdf)[0]
    num_layer = 0
    #for layer in layers:
    pan_images_data = GeoData(geo_dataframe = layers,
                   style={'color': 'green', 'fillColor': 'green', 'opacity':1.0, 'weight':1.0, 'dashArray':'2', 'fillOpacity':0.3},
                   hover_style={'fillColor': 'blue' , 'fillOpacity': 0.5},
                   visible = True,                   
                   name = 'PAN images' + str(num_layer))
    m.add(pan_images_data)
    num_layer += 1
    
    # plot the area of interest
    if (aoi_gdf_in.crs.name != 'WGS 84'):
        aoi_gdf = aoi_gdf_in.to_crs(4326, inplace=False)
    else:
        aoi_gdf = aoi_gdf_in
    aoi_data = GeoData(geo_dataframe = aoi_gdf,
                   style={'color': 'red', 'fillColor': 'red', 'opacity':1.0, 'weight':1.0, 'dashArray':'2', 'fillOpacity':0.3},
                   hover_style={'fillColor': 'blue' , 'fillOpacity': 0.5},
                   name = 'AOI')
    m.add(aoi_data)
    
    
    draw_control = DrawControl()
    draw_control.rectangle = {
       "shapeOptions": {
            "fillColor": "red",
            "color": "red",
            "fillOpacity": 0.3
        }
    }

    feature_collection = {
        'type': 'FeatureCollection',
        'features': []
    }

    def handle_draw(self, action, geo_json):
        """Do something with the GeoJSON when it's drawn on the map"""    
        feature_collection['features'].append(geo_json)

    draw_control.on_draw(handle_draw)
    control = LayersControl()
    
    m.add(draw_control)
    m.add(control)
    return m, feature_collection

def create_subsets_list(country, subregion, acquisition_day, tile_index, subset_number, geometry, crs='EPSG:3035'):
    """
    This function is used to create a GeoPandas GeoDataFrame to store all the subsets
    that have been extracted from the Copernicus VHR2015 imagery. The fields are
    country code, sub region code, acquisition day, tile index
    """
    subsets_dict = {'country': country, \
                    'subregion': subregion, \
                    'acquisition_day': acquisition_day, \
                    'tile_index': tile_index, \
                    'subset_number': subset_number, \
                    'geometry': geometry}
    
    subsets_gdf = gpd.GeoDataFrame(subsets_dict, crs=crs, index=range(0,1)) 
    return subsets_gdf

def update_subsets_list(gdf, country, subregion, acquisition_day, tile_index, subset_number, geometry, crs='EPSG:3035'):
    """
    This function adds a record to the list 
    of subsets
    """
    size = gdf.shape[0]
    countries = []
    subregions = []
    acquisition_days = []
    tiles = []
    subsets_number = []
    geometries = []
    i = 0
    for i in range(i, size):
        countries.append(gdf.iloc[i]['country'])
        subregions.append(gdf.iloc[i]['subregion'])
        acquisition_days.append(gdf.iloc[i]['acquisition_day'])
        tiles.append(gdf.iloc[i]['tile_index'])
        subsets_number.append(gdf.iloc[i]['subset_number'])
        geometries.append(gdf.iloc[i]['geometry'])
    
    countries.append(country)
    subregions.append(subregion)
    acquisition_days.append(acquisition_day)
    tiles.append(tile_index)
    subsets_number.append(subset_number)
    geometries.append(geometry)
    subsets_dict = {'country': countries, \
                    'subregion': subregions, \
                    'acquisition_day': acquisition_days, \
                    'tile_index': tiles, \
                    'subset_number': subsets_number, \
                    'geometry': geometries}
    subsets_gdf = gpd.GeoDataFrame(subsets_dict, crs=crs, index=range(0, size + 1))
    return subsets_gdf

def subset_file_path(country_code,subregion, acquisition_day, tile, subset_id_str):
    """
    This function returns the file path of a subset image
    following the structure of the imagery storage 
    <country>/<subregion>/<acquisition_day>/SUBSETS_FOLDER/<subset number>
    and the structure of the subset file naming convention
    <country>_<subregion>_<acquisition_day>_<tile>_<subset number>_<file extension>
    """
    subset_file_path = IMAGERY_FOLDER_PATH + \
                                country_code + '/' + \
                                subregion + '/' + \
                                acquisition_day + '/' + \
                                SUBSETS_FOLDER + \
                                subset_id_str + '/' + \
                                subset_file_name
    return subset_file_path

def subset_file_path(country_code,subregion, acquisition_day, tile, subset_id_str, band='PANSHARP'):
    """
    This function returns the file path of a subset image
    following the structure of the imagery storage 
    <country>/<subregion>/<acquisition_day>/SUBSETS_FOLDER/<subset number>
    and the structure of the subset file naming convention
    <country>_<subregion>_<acquisition_day>_<tile>_<subset number>_<file extension>
    """
    if (band == 'MS'):
        file_ext = 'ms.tif'
    elif (band == 'PAN'):
        file_ext = 'pan.tif'
    elif (band == 'PAN_MODIFIED'):
        file_ext = 'sharp_modified.tif'
    else:
        file_ext = 'sharp.tif'
        
    subset_file_name = country_code + '_' + \
                       subregion + '_' + \
                       acquisition_day + '_' + \
                       tile + '_' + \
                       subset_id_str + '_' + \
                       file_ext

    subset_file_path = IMAGERY_FOLDER_PATH + \
                                country_code + '/' + \
                                subregion + '/' + \
                                acquisition_day + '/' + \
                                SUBSETS_FOLDER + \
                                subset_id_str + '/' + \
                                subset_file_name
    return subset_file_path

def patch_file_path(country_code,subregion, acquisition_day, tile, subset_id_str, patch_index_x, patch_index_y):
    """
    This function returns the file path of a patch image
    following the structure of the imagery storage 
    <country>/<subregion>/<acquisition_day>/PATCHES_FOLDER/<subset number>
    and the structure of the subset file naming convention
    <country>_<subregion>_<acquisition_day>_<tile>_<subset number>_<file extension>
    """
    file_ext = str(patch_index_y) + '_' + str(patch_index_x) + '.tif'
    
    patch_file_name = country_code + '_' + \
                       subregion + '_' + \
                       acquisition_day + '_' + \
                       tile + '_' + \
                       subset_id_str + '_' + \
                       file_ext

    patch_file_path = IMAGERY_FOLDER_PATH + \
                                country_code + '/' + \
                                subregion + '/' + \
                                acquisition_day + '/' + \
                                PATCHES_FOLDER + \
                                subset_id_str + '/' + \
                                patch_file_name
    return patch_file_path

def sub_folder_path(country_code,subregion, acquisition_day, tile, subset_id_str, image_content_type = 'SUBSET'):
    """
    This function returns the path of the folder that contains the image subsets or patches
    or footprints. The path follows the structure of the imagery storage 
    <country>/<subregion>/<acquisition_day>/PATCHES_FOLDER/<subset number>/
    """
    if (image_content_type == 'PATCH'):
        sub_folder = PATCHES_FOLDER
    elif (image_content_type == 'FOOTPRINT'):
        sub_folder = FOOTPRINTS_FOLDER
    else:
        sub_folder = SUBSETS_FOLDER
    
    patch_folder_path = IMAGERY_FOLDER_PATH + \
                                country_code + '/' + \
                                subregion + '/' + \
                                acquisition_day + '/' + \
                                sub_folder + \
                                subset_id_str + '/'
    return patch_folder_path

def footprints_folder_path(country_code,subregion, acquisition_day, tile, subset_id_str):
    """
    This function returns the path of the folder that contains the footprints
    extracted from an image patch. The path follows the structure of the imagery storage 
    <country>/<subregion>/<acquisition_day>/PATCHES_FOLDER/<subset number>/
    """
    footprints_folder_path = IMAGERY_FOLDER_PATH + \
                                country_code + '/' + \
                                subregion + '/' + \
                                acquisition_day + '/' + \
                                FOOTPRINTS_FOLDER + \
                                subset_id_str + '/'
    return footprints_folder_path

def process_patches(patches_path, osm_data, footprints_path):
    """
    This function extracts the building footprints images from
    all the patches of a subset image in the patches folder
    """
    patches_list = []
    for file in os.listdir(patches_path):
        if file.endswith(".tif"):
            patches_list.append(os.path.join(patches_path, file))
    footprints_list = []
    for patch_path in patches_list:
        #print(patch_path)
        patch_index_str = patch_path[-7:-4]
        img_list = extract_building_images(patch_path, patch_index_str, osm_data, footprints_path)
        for img in img_list:
            footprints_list.append(img)

    return footprints_list

def create_gdf_list(collection_gdf):
    '''
    This function returns a list of geodataframes of 
    one single polygon from a geodataframe containing
    a number of polygons
    '''
    col_names = []
    for col in collection_gdf.columns:
        col_names.append(col)
    
    layers = []
    num_cols = len(col_names)
    for index, row in collection_gdf.iterrows():
        image_dict = dict(zip(col_names, row))
        layer = gpd.GeoDataFrame(image_dict, crs=collection_gdf.crs, index=range(0,1))
        layers.append(layer)
    return layers

def create_gdf(pd_series, crs):
    '''
    This function returns a geodataframe of one single geometry
    from a Pandas series extracted as a row of a geodataframe 
    '''
    col_names = []
    for col in pd_series.index:
        col_names.append(col)
    
    values = pd_series.values
    num_cols = len(col_names)
    image_dict = dict(zip(col_names, values))
    layer = gpd.GeoDataFrame(image_dict, crs=crs, index=range(0,1))
    return layer

def select_pan_images(vhr_pan_gdf, bbox_gdf):
    '''
    This function selects the panchromatic image that covers most, if not all, the 
    subset of the area of interest 
    '''
    select_images_gdf = None
    vhr_pan_gdf_intersect_list = vhr_pan_gdf.intersects(bbox_gdf.loc[0, 'geometry'])
    print('Number of PAN images that intersect our area of interest: {0:d}'.format(sum(vhr_pan_gdf_intersect_list)))
    
    vhr_pan_gdf_cover_list = vhr_pan_gdf.covers(bbox_gdf.loc[0, 'geometry'])
    print('Number of PAN images that cover our area of interest: {0:d}'.format(sum(vhr_pan_gdf_cover_list)))

    count = 1
    pan_image_rel_path_list = []
    if (vhr_pan_gdf_cover_list.any()):
        vhr_pan_cover_gdf = vhr_pan_gdf[vhr_pan_gdf_cover_list]
        for path in vhr_pan_cover_gdf['relpath']:
            pan_image_rel_path_list.append(path)
            print('Panchromatic image covering the area of interest:\n{0:d}) {1:s}'.format(count, path))
            count += 1
        select_images_gdf = vhr_pan_cover_gdf
    elif (vhr_pan_gdf_intersect_list.any()):
        vhr_pan_intersect_gdf = vhr_pan_gdf[vhr_pan_gdf_intersect_list]
        for path in vhr_pan_intersect_gdf['relpath']:
            pan_image_rel_path_list.append(path)
            print('Panchromatic image intersecting the area of interest:\n{0:d}) {1:s}'.format(count, path))
            count += 1
        select_images_gdf = vhr_pan_intersect_gdf
    else:
        print('There are no images that cover or intersect the area of interest')
    return select_images_gdf

def tif_to_png(tif_file_path):
    '''
    This function creates a new PNG file from a TIF file. 
    The name of the PNG file will be the same as that of the TIF file
    with the .png extension. If the PNG file already exists the function
    does not return anything.
    '''
    success_png_file_path = None
    tif_file = Path(tif_file_path)
    png_file_path = tif_file_path[:-3] + 'png'
    #print('PNG file: {:s}'.format(png_file_path))
    png_file = Path(png_file_path)
    
    if (tif_file.is_file() and tif_file_path.endswith('.tif') and not png_file.is_file()):
        # source tif image
        img_ds = gdal.Open(tif_file_path)
        img_cols = img_ds.RasterXSize
        img_rows = img_ds.RasterYSize
        min_size = min(img_cols, img_rows)
        #print('Min tif image size: {:d}'.format(min_size))
        img_origin_x, img_resolution_x, img_row_rotation, img_origin_y, img_col_rotation, img_resolution_y = img_ds.GetGeoTransform()
        
        # target png image
        img_origin_col = 0
        img_origin_row = 0
        img_std_rows = min_size
        img_std_cols = min_size
        if (img_rows >= img_std_rows and img_cols >= img_std_cols): # check the size of the source image
            source_img_bands = getRasterBands(img_ds, img_origin_col, img_origin_row, img_std_cols, img_std_rows)
    
            target_img_data_bands = []
            for i in range(0, len(source_img_bands)):
                p = 255 * (source_img_bands[i] / source_img_bands[i].max())
                target_img_data_bands.append(np.rint(p))
        
            bands_stack = np.dstack([target_img_data_bands[0], target_img_data_bands[1], target_img_data_bands[2]]).astype(np.uint8)
            stack_img = Image.fromarray(bands_stack, 'RGB')
            success_png_file_path = tif_file_path[:-3] + 'png'
            stack_img.save(png_file_path, 'PNG')
       # else:
            #print('The size of the source image {:d}x{:d} is less than the standard size target image (64x64)'.format(img_cols, img_rows))
            
    #else:
        #print('The source image {:s} does not exist or is not a TIF file, or the target PNG file already exists.'.format(tif_file_path))
        
    return success_png_file_path

def copy_split_files(source_dir, target_dir, num_files_train_val_test_dict):
    '''
    This function copies the data files from the source dataset, that contains one subfolder
    for each category, to the target directory that contains two levels of subfolders. The first
    level is train, validation, and test, and the second level is for the category subfolders
    within each train, validation, and test folders. The files in the target directory are distributed
    into train, validation and test according to the intervals passed in a dictionary as argument 
    '''
    source_dir_path = Path(source_dir)
    source_subfolders_list = [os.path.basename(sub_folder) for sub_folder in source_dir_path.iterdir() if sub_folder.is_dir()]
    target_dir_path = Path(target_dir)
    target_subfolders_list = [os.path.basename(sub_folder) for sub_folder in target_dir_path.iterdir() if sub_folder.is_dir()]
    datasets_index = ['train', 'validation', 'test']
    for target_subfolder in target_subfolders_list:
        print('\nDataset: {:s}'.format(target_subfolder))
        for category_index in range(len(source_subfolders_list)):
            category = source_subfolders_list[category_index]
            print('Rooftop type: {:s}'.format(category))
            source_category_dir_path = Path(os.path.join(source_dir_path, category))
            target_category_dir_path = Path(os.path.join(str(target_dir_path) + '/' + target_subfolder, category))
            print('Source category directory: {:s}'.format(str(source_category_dir_path)))
            print('Target category directory: {:s}'.format(str(target_category_dir_path)))
            #cat_index = source_subfolders_list[datasets_index]
            #print(target_subfolder)
            #print(num_files_train_val_test_dict[target_subfolder])
            index = num_files_train_val_test_dict[target_subfolder]
            start = index[0]
            end = index[1]
            footprints = [os.path.basename(file) for file in source_category_dir_path.iterdir() if file.is_file()]
            #print(start, end)
            num_files_copied = 0
            for k in range(start, end):
                img_tif = footprints[k]
                source_file_path = Path(os.path.join(source_category_dir_path, img_tif))
                target_file_path = Path(os.path.join(target_category_dir_path, img_tif))
                shutil.copyfile(source_file_path, target_file_path)
                num_files_copied += 1
            print('Number of files copied to target subdirectory {:s}: {:d}'.format(category, num_files_copied))
            

def split_category_data(num_train, num_val, num_test):
    '''
    This function returns a dictionary with the intervals
    of data points (indexes) to be included in the train, validation,
    and test set for a category
    '''
    train_start = 0
    train_end = num_train #3000
    num_images_for_validation = num_val #1000
    num_images_for_test = num_test #800
    validation_start = train_end
    validation_end = validation_start + num_images_for_validation
    test_start = validation_end
    test_end = test_start + num_images_for_test
    train_interval = [train_start, train_end]
    val_interval = [validation_start, validation_end]
    test_interval = [test_start, test_end]
    num_files_train_val_test = [train_interval,
                                val_interval,
                                test_interval]
    num_files_train_val_test_dict = {'train': train_interval, 
                                     'validation': val_interval,
                                     'test': test_interval}
    return num_files_train_val_test_dict

def compute_class_membership(predictions): # deprecated
    '''
    This function maps the highest probability in the predictions
    to a class index. Each prediction is an array with the probabilities
    of a data point to be a member of one class. For example, if a prediction 
    contains three probabilities the function returns an integer that represents
    the class index with the highest probability value.
    The same result can be achieved using the NumPy function
    np.argmax (predictions, axis = 1)
    '''
    class_membership = []
    prediction_index = 0
    for prediction in predictions:
        max_probability = 0.0
        max_c = 0  
        for c in range(0,3):
            if (prediction[c] > max_probability):
                max_probability = prediction[c]
                max_c = c
        class_membership.append(max_c)
    return np.array(class_membership)

def confusion_data(best_model, test_dataset):
    '''
    This function returns the predictions from the model passed
    as argument, assumed to be the best achieved, and the corresponding
    labels from the test set in order to be used to compute the 
    confusion matrix for the evaluation of the multi-class classification
    '''
    test_predictions = np.array([])
    test_labels = np.array([])
    for batch_imgs, batch_labels in test_dataset:
        batch_predictions = np.argmax(best_model.predict(batch_imgs, verbose=3), axis = 1)
        test_predictions = np.concatenate((test_predictions, batch_predictions))
        test_labels = np.concatenate((test_labels, batch_labels))
    return test_predictions, test_labels