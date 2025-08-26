Copernicus Sentinel-2
=====================
This repository contains notebooks and Python scripts to process Sentinel-2 datasets. The datasets can be download fromt the [Copernicus Data Space ecosystem](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-data/sentinel-2). A Sentinel-2 dataset contains 13 bands in JPG files, from visible to IF. One task would be

1. Extract patches from a Sentinel-2 tile
2. Create GeoTIFF patches buy stacking the RGB bands with the coordinates in one GeoTIFF file