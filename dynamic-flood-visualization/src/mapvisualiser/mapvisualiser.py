import dask
import hvplot.xarray
import numpy as np
import typing
from typing import Dict, Union, Any
from bokeh.models import ColorMapper, LinearColorMapper, LogColorMapper, CategoricalColorMapper
from bokeh.transform import transform
from bokeh.palettes import Blues8, Reds8, Viridis8, Plasma8, Inferno8
# For type hints - import the actual types  
from bokeh.core.property.dataspec import DataSpec
from dcloader.loader import DcLoader
from typing import List

class MapVisualiser:
    def __init__(self, dcloader_instance: DcLoader):
        """
        Initializes the visualizer with a completed DcLoader instance.

        Args:
            dcloader_instance (DcLoader): An instance of DcLoader with data
                                          already loaded.
        """
        if dcloader_instance.dc is None:
            raise ValueError("The provided DcLoader instance must have data loaded first.")
        if dcloader_instance.dc.data_vars is None:
            raise ValueError("The provided DcLoader instance must have data variables loaded first.")
        if  not np.array(['exclusion_mask', 'reference_water_mask', 'advisory_flags']) in  dcloader_instance.dc.data_vars.keys() :
            raise ValueError("The provided DcLoader instance must have the required data variables loaded.")
        
        self.dc = dcloader_instance.dc
        self.original_crs = dcloader_instance.crs
        self.bounding_box = dcloader_instance.bounding_box
        self.refined_data = {}
        self.plottable_data = {}
        self.flood_variables = [var for var in self.dc.data_vars.keys() if var not in ['exclusion_mask', 'reference_water_mask', 'advisory_flags']] 

    def select_and_refine_variable(self, variable_name: str, time_index: int = 0):
        """
        Selects a variable, applies exclusion and water masks to refine it.

        Args:
            variable_name (str): The name of the flood extent variable to process
                                 (e.g., 'tuw_flood_extent').
            time_index (int): The time index to select from the data cube.
        """
        print(f"Selecting and refining variable '{variable_name}'...")
        
        if variable_name not in self.dc:
            raise ValueError(f"Variable '{variable_name}' not found in the dataset. "
                             f"Available variables are: {list(self.dc.data_vars)}")

        time_slice = self.dc.isel(time=time_index)
        
        flood_extent = time_slice[variable_name]
        exclusion_mask = time_slice['exclusion_mask']
        reference_water_mask = time_slice['reference_water_mask']
        
        # A pixel is considered flooded if the model output is 1 (flood)
        # AND it's not in an excluded region AND it's not permanent water.
        self.refined_data[variable_name] = flood_extent.where(
            (flood_extent > 0) & 
            (exclusion_mask == 0) & 
            (reference_water_mask == 0)
        )
        print("Variable refined successfully.")

    def select_and_refine_all_vars(self):
        for vars in self.flood_variables:
            self.select_and_refine_variable(vars)

    def prepare_for_map_overlay(self, variable_name: str):
        """
        Reprojects and clips the refined data to prepare it for map overlay.
        """
        if variable_name not in self.refined_data:
            raise RuntimeError("Run select_and_refine_variable() first to generate refined data.")
            
        print("Reprojecting data for map overlay...")
        
        # Add the original CRS info to the DataArray
        data_with_crs = self.refined_data[variable_name].rio.write_crs(self.original_crs)
        
        # Reproject to WGS84 (EPSG:4326) for web mapping
        reprojected_data = data_with_crs.rio.reproject("EPSG:4326")
        
        # Clip to the precise bounding box to ensure clean edges
        min_lon, min_lat, max_lon, max_lat = self.bounding_box
        self.plottable_data[variable_name] = reprojected_data.rio.clip_box(
            minx=min_lon, maxx=max_lon, miny=min_lat, maxy=max_lat
        )
        print("Data prepared for plotting.")

    def plot_on_map(self, cmap: list, alpha: float = 0.1, title: str = None, tiles: str = "OSM"):
        """
        Generates an interactive map by overlaying the plottable data.

        Args:
            cmap (list): A list of colors for the colormap.
            alpha (float): The transparency of the overlay (0.0 to 1.0).
            title (str, optional): The title for the map. Defaults to a generated title.
            tiles (str): The map tile provider (e.g., 'OSM', 'EsriImagery').

        Returns:
            hvplot object: An interactive map visualization.
        """
        if self.plottable_data is None:
            raise RuntimeError("Run prepare_for_map_overlay() first to get plottable data.")

        print("Generating interactive map...")

        if title is None:
            date = self.dc.time.isel(time=0).dt.strftime('%Y-%m-%d').item()
            title = f"Flood Extent for '{self.variable_name}' on {date}"

        flood_map = self.plottable_data.hvplot.image(
            x='x', y='y',
            geo=True,
            tiles=tiles,
            cmap=cmap,
            alpha=alpha,
            frame_width=800,
            frame_height=600,
            title=title,
            xlabel="Longitude",
            ylabel="Latitude",
            colorbar=False
        )
        return flood_map