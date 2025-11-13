import pystac_client
from dask.distributed import wait
from dask.distributed import Client, wait
import pyproj
import rioxarray


import rioxarray

class DataPreprocessor:
    """
    Handles preprocessing of geospatial flood data such as CRS assignment,
    reprojection, and spatial clipping.
    """

    def __init__(self, dc):
        """
        Args:
            dc (xarray.Dataset): The dataset containing geospatial data bands.
        """
        self.dc = dc

    
    def clean_flood_band(self, band_name):
        """
        Cleans bands  by replacing invalid values (255) with NaN 

        Args:
            band_name (str): The name of the band to clean.

        Returns:
            xarray.DataArray: Cleaned band DataArray.
        """
        if self.dc is None:
            raise RuntimeError("Data has not been loaded yet. Call load_GFM_data() first.")

        if band_name in self.dc:
            self.dc[band_name] = self.dc[band_name].where(self.dc[band_name] != 255).compute()
            return self.dc[band_name]
        else:
            available_bands = list(self.dc.data_vars)
            raise ValueError(f"Band '{band_name}' not found. Available bands are: {available_bands}")
        
    def limit_time_Of_band(self, band_name,timestamp):
        
        self.dc[band_name] = self.dc[band_name].isel(time=timestamp)
        return self.dc[band_name]
        
        

    def clip_to_bbox(self, da, bbox):
        """
        Clips a DataArray to a bounding box.

        Args:
            da (xarray.DataArray): DataArray to clip.
            bbox (tuple): Bounding box (minlon, maxlon, minlat, maxlat).

        Returns:
            xarray.DataArray: Clipped DataArray.
        """
        minlon, maxlon, minlat, maxlat = bbox
        return da.rio.clip_box(minx=minlon, maxx=maxlon, miny=minlat, maxy=maxlat)
    
    def reproject_band(self, band, bbox ):
        """
        Reproject bands to WGS84 (lon/lat)

        Args:
            band (str): The name of the band to clean.

        Returns:
            xarray.DataArray: Cleaned band DataArray.
        """
        crs = self.dc[band].rio.crs
        ae_crs  =crs.to_proj4()

        da = self.dc[band].rio.write_crs(ae_crs)

        da = da.rio.reproject("EPSG:4326")

        da = self.clip_to_bbox(da, bbox)

        return da

    
    
