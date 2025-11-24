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
import matplotlib.pyplot as plt
import rioxarray as rio
import holoviews as hv
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape, Polygon, MultiPolygon
import hvplot.pandas  # ← This enables hvplot for GeoDataFrames!
from tqdm.auto import tqdm
import pickle
import os
import xarray as xr
from datetime import datetime
import re
from color_schemes import COLOR_PALETTES

class MapVisualiser:
    
    PossibleModels = {"dlr","lst","tuw","ensemble"}

    def __init__(self, dcloader_instance: DcLoader = None, load_from_file: str = None):
        """
        Initializes the visualizer with a completed DcLoader instance.

        Args:
            dcloader_instance (DcLoader): An instance of DcLoader with data
                                          already loaded.
        """
        if load_from_file is not None and dcloader_instance is not None:
            raise ValueError("Provide either 'dcloader_instance' or 'load_from_file', not both.")
        
        if load_from_file is None and dcloader_instance is None:
            raise ValueError("Either 'dcloader_instance' or 'load_from_file' must be provided.")
        
        if load_from_file is not None:
            # Load from file using the static method
            loaded_visualiser = MapVisualiser.build_from_file(load_from_file)
            
            # Copy all attributes from loaded visualiser
            self.dc = loaded_visualiser.dc
            self.original_crs = loaded_visualiser.original_crs
            self.bounding_box = loaded_visualiser.bounding_box
            self.refined_data = loaded_visualiser.refined_data
            self.plottable_data = loaded_visualiser.plottable_data
            self.polygons_data = loaded_visualiser.polygons_data
            self.flood_variables = loaded_visualiser.flood_variables
        else:
            # Initialize with DcLoader
            if dcloader_instance.dc is None:
                raise ValueError("The provided DcLoader instance must have data loaded first.")
            if dcloader_instance.dc.data_vars is None:
                raise ValueError("The provided DcLoader instance must have data variables loaded first.")

            req_vars = ['exclusion_mask', 'reference_water_mask', 'advisory_flags']
            for var in req_vars:
                if var not in dcloader_instance.dc.data_vars.keys():
                    raise ValueError(f"The provided DcLoader instance must have the required data variables loaded: {var}")
            
            self.dc = dcloader_instance.dc
            self.original_crs = dcloader_instance.crs
            self.bounding_box = dcloader_instance.bounding_box
            self.refined_data = {}
            self.plottable_data = {} # here we store the reprojected and clipped dataarrays for mapping 
            self.polygons_data = {}
            self.flood_variables = [var for var in self.dc.data_vars.keys() if var not in req_vars]

    def __str__(self):
        return f"""Bounding Box: {self.bounding_box}
        Flood variables: {len(self.flood_variables)}
        Refined variables: {len(self.refined_data)} 
        Plottable variables: {len(self.plottable_data)}
        Polygon variables: {len(self.polygons_data)}"""

    @staticmethod
    def remove_all_holes(geom):
        if geom.geom_type == "Polygon":
            return Polygon(geom.exterior)  # rebuild without interiors
        elif geom.geom_type == "MultiPolygon":
            return type(geom)([Polygon(p.exterior) for p in geom.geoms])
        else:
            return geom
        
    @staticmethod
    def get_optimal_projected_crs(lon: float, lat: float, bounds: tuple = None) -> str:
        """
        Automatically determines the optimal projected CRS for given coordinates.
        Uses UTM for most regions, with special handling for Europe.
        
        Args:
            lon (float): Center longitude in degrees
            lat (float): Center latitude in degrees
            bounds (tuple, optional): Bounding box (minx, miny, maxx, maxy)
        
        Returns:
            str: EPSG code for the appropriate projected CRS
        """
        # European regions: Use ETRS89 / LAEA Europe for better accuracy across countries
        # Covers: 10°W to 40°E, 35°N to 72°N (most of Europe including Germany, Greece)
        if -10 <= lon <= 40 and 35 <= lat <= 72:
            # ETRS89-extended / LAEA Europe - ideal for Europe-wide projects
            return "EPSG:3035"
        
        # For regions outside Europe, use UTM
        utm_zone = int((lon + 180) / 6) + 1
        
        # Determine hemisphere
        hemisphere = 'north' if lat >= 0 else 'south'
        
        # Construct EPSG code
        if hemisphere == 'north':
            epsg_code = f"EPSG:{32600 + utm_zone}"
        else:
            epsg_code = f"EPSG:{32700 + utm_zone}"
        
        return epsg_code

    @staticmethod
    def get_region_name(lon: float, lat: float) -> str:
        """
        Returns a human-readable region name for logging purposes.
        
        Args:
            lon (float): Longitude
            lat (float): Latitude
        
        Returns:
            str: Region name
        """
        # Pakistan region
        if 60 <= lon <= 80 and 20 <= lat <= 40:
            return "Pakistan/South Asia"
        # Greece region
        elif 19 <= lon <= 30 and 34 <= lat <= 42:
            return "Greece"
        # Germany region
        elif 5 <= lon <= 16 and 47 <= lat <= 56:
            return "Germany"
        # General Europe
        elif -10 <= lon <= 40 and 35 <= lat <= 72:
            return "Europe"
        # Asia
        elif 60 <= lon <= 150 and -10 <= lat <= 60:
            return "Asia"
        else:
            return "Unknown"
        
    @staticmethod
    def build_from_file(filepath: str) -> 'MapVisualiser':
        """
        Loads a MapVisualiser from a saved file (static method).
        
        Args:
            filepath (str): Path to the .mapvis index file or base path.
        
        Returns:
            MapVisualiser: A new MapVisualiser instance with loaded data.
        
        Raises:
            FileNotFoundError: If the specified file doesn't exist.
        """
        # Handle different file path formats
        if not filepath.endswith('.mapvis'):
            filepath = f"{filepath}.mapvis"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MapVisualiser file not found: {filepath}")
        
        print(f"Loading MapVisualiser from {filepath}...")
        
        # Load index
        with open(filepath, 'r') as f:
            import json
            index = json.load(f)
        
        base_path = index['base_path']
        files = index['files']
        
        # Create progress bar
        total_steps = 4 + len(files.get('polygons', {}))
        pbar = tqdm(total=total_steps, desc="Loading MapVisualiser", unit="step")
        
        # Create a new instance (bypass __init__)
        visualiser = object.__new__(MapVisualiser)
        
        # 1. Load metadata
        pbar.set_postfix_str("Loading metadata")
        with open(files['metadata'], 'rb') as f:
            metadata = pickle.load(f)
        
        visualiser.original_crs = metadata['original_crs']
        visualiser.bounding_box = metadata['bounding_box']
        visualiser.flood_variables = metadata['flood_variables']
        pbar.update(1)
        
        # 2. Load datacube
        pbar.set_postfix_str("Loading datacube")
        visualiser.dc = xr.open_dataset(files['datacube'])
        pbar.update(1)
        
        # 3. Load refined_data
        pbar.set_postfix_str("Loading refined data")
        visualiser.refined_data = {}
        if files['refined'] and os.path.exists(files['refined']):
            refined_ds = xr.open_dataset(files['refined'])
            visualiser.refined_data = {var: refined_ds[var] for var in metadata['refined_variables']}
        pbar.update(1)
        
        # 4. Load plottable_data
        pbar.set_postfix_str("Loading plottable data")
        visualiser.plottable_data = {}
        if files['plottable'] and os.path.exists(files['plottable']):
            plottable_ds = xr.open_dataset(files['plottable'])
            visualiser.plottable_data = {var: plottable_ds[var] for var in metadata['plottable_variables']}
        pbar.update(1)
        
        # 5. Load polygons_data
        visualiser.polygons_data = {}
        if files['polygons']:
            pbar.set_postfix_str("Loading polygons")
            for var_name, poly_file in files['polygons'].items():
                if os.path.exists(poly_file):
                    visualiser.polygons_data[var_name] = gpd.read_file(poly_file)
                pbar.update(1)
        
        pbar.close()
        
        print(f"✓ MapVisualiser loaded successfully")
        print(f"  CRS: {visualiser.original_crs}")
        print(f"  Bounding box: {visualiser.bounding_box}")
        print(f"  Flood variables: {len(visualiser.flood_variables)}")
        print(f"  Refined variables: {len(visualiser.refined_data)}")
        print(f"  Plottable variables: {len(visualiser.plottable_data)}")
        print(f"  Polygon variables: {len(visualiser.polygons_data)}")
        print(f"  Saved on: {metadata['save_timestamp']}")
        
        return visualiser

    @staticmethod
    def _validate_expression_syntax(expression: str) -> bool:
        """Validates expression safety"""
        dangerous = [
            r'__\w+__',
            r'import\s',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'compile\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
        ]
        return not any(re.search(p, expression) for p in dangerous)
    
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
            (flood_extent != 255) & 
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
        print(f"{variable_name} prepared for plotting.")

    def prepare_for_map_overlay_all_vars(self):
        for vars in self.flood_variables:
            self.prepare_for_map_overlay(vars)

    def build_polygons(self, variable_name: str, smooth_distance: float = 0.0001, threshold: float = 0.0, save_to_polygons: bool = False) -> gpd.GeoDataFrame:
        """
        Converts raster flood data to smooth vector polygons with holes filled.
        
        Args:
            variable_name (str): The name of the variable to convert to polygons.
            smooth_distance (float): The buffer distance for smoothing. Adjust based on your CRS.
                                    Default is 0.0001 (suitable for WGS84 degrees).
        
        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame containing the flood polygons.
        """
        if variable_name not in self.plottable_data:
            raise RuntimeError(f"Variable '{variable_name}' not found in plottable_data. "
                             f"Run prepare_for_map_overlay('{variable_name}') first.")
        
        print(f"Building polygons for '{variable_name}'...")
        
        # Get the plottable data
        variable_mat = self.plottable_data[variable_name].where(self.plottable_data[variable_name] > threshold)
        
        # Create binary mask for flood extent
        mask = (variable_mat > threshold)
        
        # Convert raster to vector polygons
        shapes_gen = features.shapes(
            mask.astype(np.uint8).values,
            mask=~np.isnan(variable_mat.values),
            transform=variable_mat.rio.transform()
        )

        # Extract only polygons where mask > 0
        polygons = [shape(geom) for geom, val in shapes_gen if val > threshold]
        
        if not polygons:
            print(f"Warning: No flood polygons found for '{variable_name}'")
            # Return empty GeoDataFrame with proper CRS
            gdf = gpd.GeoDataFrame(geometry=[], crs=self.original_crs)
            self.polygons_data[variable_name] = gdf
            return gdf
        
        # Build GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=self.original_crs)
        
        # Fill all holes
        print(f"  Filling holes in polygons...")
        gdf["geometry"] = gdf["geometry"].apply(self.remove_all_holes)
        
        print(f"  Smoothing polygons...")        
        # Check if CRS is geographic (uses degrees)
        if self.original_crs.is_geographic:
            # Get center point of bounding box for CRS selection
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            
            # Get appropriate projected CRS and region name
            projected_crs = self.get_optimal_projected_crs(center_lon, center_lat, bounds)
            region_name = self.get_region_name(center_lon, center_lat)
            
            print(f"    Region detected: {region_name}")
            print(f"    Converting from {self.original_crs} to {projected_crs} for accurate buffering...")

            # Reproject to projected CRS
            gdf_projected = gdf.to_crs(projected_crs)
            
            # Convert smooth_distance from degrees to meters
            # Approximate: 1 degree ≈ 111 km at equator
            # Adjust for latitude: actual_distance = 111km * cos(latitude)
            lat_factor = np.cos(np.radians(center_lat))
            smooth_distance_meters = smooth_distance * 111000 * lat_factor
            
            print(f"    Using buffer distance: {smooth_distance_meters:.1f} meters")
            
            # Apply buffer operations in projected CRS
            gdf_projected["geometry"] = gdf_projected.buffer(0)  # repair topology
            gdf_projected = gdf_projected.explode(ignore_index=True)
            gdf_projected["geometry"] = gdf_projected.buffer(smooth_distance_meters).buffer(-smooth_distance_meters)
            
            # Reproject back to original CRS (WGS84) for web mapping
            gdf = gdf_projected.to_crs(self.original_crs)

            print(f"    Converted back to {self.original_crs}")
        else:
            # CRS is already projected, use smooth_distance as-is
            print(f"    Using projected CRS, buffer distance: {smooth_distance} units")
            gdf["geometry"] = gdf.buffer(0)
            gdf = gdf.explode(ignore_index=True)
            gdf["geometry"] = gdf.buffer(smooth_distance).buffer(-smooth_distance)

        if save_to_polygons:
            self.polygons_data[variable_name] = gdf

        print(f"Polygons built successfully for '{variable_name}'. Found {len(gdf)} polygon(s).")
        return gdf
    
    def build_polygons_all_vars(self, smooth_distance: float = 0.0001):
        """
        Builds polygons for all plottable variables.
        
        Args:
            smooth_distance (float): The buffer distance for smoothing. Adjust based on your CRS.
                                    Default is 0.0001 (suitable for WGS84 degrees).
        """
        if not self.plottable_data:
            raise RuntimeError("No plottable data available. Run prepare_for_map_overlay_all_vars() first.")
        
        print(f"Building polygons for all {len(self.plottable_data)} plottable variables...")
        
        for var_name in self.plottable_data.keys():
            self.build_polygons(var_name, smooth_distance=smooth_distance)
        
        print(f"All polygons built successfully. Total: {len(self.polygons_data)} variable(s).")
    
    def apply_plottable(self, expression: str, output_name: str = None, save_to_plottable: bool = False) -> xr.DataArray:
        """
        Applies a mathematical expression element-wise on plottable data variables.
        
        Supports basic operations: +, -, *, /, //, %, ** (power)
        Also supports parentheses for operation precedence.
        
        Args:
            expression (str): Mathematical expression using variable names from plottable_data.
                            Example: "tuw_likelihood * tuw_flood_extent"
                            Example: "(tuw_likelihood + 0.5) * tuw_flood_extent"
            output_name (str, optional): Name for the resulting DataArray. 
                                        Defaults to the expression string.
        
        Returns:
            xr.DataArray: Result of the mathematical operation.
        
        Raises:
            ValueError: If variables don't exist or have incompatible shapes.
            SyntaxError: If expression contains invalid operations.
        
        Examples:
            >>> # Multiply two variables
            >>> result = visualiser.apply_plottable("tuw_likelihood * tuw_flood_extent")
            
            >>> # Complex expression
            >>> result = visualiser.apply_plottable("(tuw_likelihood + 0.5) * tuw_flood_extent / 2")
            
            >>> # Save result to plottable_data
            >>> result = visualiser.apply_plottable("tuw_likelihood * tuw_flood_extent", 
            ...                                      output_name="combined_risk")
            >>> visualiser.plottable_data["combined_risk"] = result
        """
        if not self.plottable_data:
            raise RuntimeError("No plottable data available. Run prepare_for_map_overlay_all_vars() first.")
        
        print(f"Parsing expression: '{expression}'")
        
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        potential_vars = re.findall(var_pattern, expression)
        
        # Filter out Python keywords and numeric literals
        python_keywords = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'}
        variables_in_expr = [v for v in potential_vars if v not in python_keywords]
        
        # Remove duplicates while preserving order
        variables_in_expr = list(dict.fromkeys(variables_in_expr))        
        # Validate that all variables exist in plottable_data
        missing_vars = [v for v in variables_in_expr if v not in self.plottable_data]
        if missing_vars:
            available = list(self.plottable_data.keys())
            raise ValueError(
                f"Variables not found in plottable_data: {missing_vars}\n"
                f"Available variables: {available}"
            )
        
        if variables_in_expr:
            reference_var = variables_in_expr[0]
            reference_shape = self.plottable_data[reference_var].shape
            reference_coords = self.plottable_data[reference_var].coords
            
            # Validate that all variables have the same shape
            print(f"  Validating shapes (reference: {reference_shape})...")
            for var_name in variables_in_expr:
                var_shape = self.plottable_data[var_name].shape
                if var_shape != reference_shape:
                    raise ValueError(
                        f"Shape mismatch: '{var_name}' has shape {var_shape}, "
                        f"but '{reference_var}' has shape {reference_shape}. "
                        f"All variables must have the same dimensions."
                    )
            print(f"  ✓ All variables have compatible shapes")
        
        namespace = {
            var_name: self.plottable_data[var_name] 
            for var_name in variables_in_expr
        }
        
        # Validate expression safety (no dangerous operations)
        # Allow only mathematical operators and parentheses
        safe_chars = set('0123456789+-*/()%. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
        if not all(c in safe_chars for c in expression.replace(' ', '')):
            raise SyntaxError(
                f"Expression contains invalid characters. "
                f"Only alphanumeric, operators (+, -, *, /, //, %, **), and parentheses are allowed."
            )
        
        # Evaluate the expression
        print(f"  Evaluating expression...")
        try:
            result = eval(expression, {"__builtins__": {}}, namespace)
        except NameError as e:
            raise ValueError(f"Invalid variable in expression: {e}")
        except Exception as e:
            raise SyntaxError(f"Error evaluating expression: {e}")
        
        # Ensure result is a DataArray
        if not isinstance(result, xr.DataArray):
            # If result is a scalar, broadcast it to match the reference shape
            if variables_in_expr:
                result = xr.DataArray(
                    data=np.full(reference_shape, result),
                    coords=reference_coords,
                    dims=self.plottable_data[reference_var].dims
                )
            else:
                raise ValueError("Expression must contain at least one variable from plottable_data")
        
        if output_name is None:
            output_name = expression
        result.name = output_name

        if save_to_plottable:
            self.plottable_data[output_name] = result
            print(f"  Saved to plottable_data['{output_name}']")
        
        print(f"✓ Expression evaluated successfully")
        print(f"  Result shape: {result.shape}")
        print(f"  Result name: '{result.name}'")
        
        return result

    def plot_refined_data_grid(self, variables:list=None, cmap=COLOR_PALETTES['light_to_strong_blue'], figsize=(20, 20), sample_rate=0):
        """
        Plots all refined data variables in a grid layout using matplotlib.
        
        Args:
            cmap (str): The colormap to use for plotting.
            figsize (tuple): The figure size (width, height).
            sample_rate (int): Sample every nth pixel for faster plotting.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if not self.refined_data:
            raise RuntimeError("No refined data available. Run select_and_refine_all_vars() first.")
        
        # Limit to first 16 variables
        if variables is None:
            variables = list(self.refined_data.keys())[:16]
        else:
            variables = [var for var in variables if var in self.refined_data][:16]
        n_vars = len(variables)
        
        # Calculate grid dimensions
        if n_vars <= 4:
            n_rows, n_cols = 2, 2
        elif n_vars <= 9:
            n_rows, n_cols = 3, 3
        else:
            n_rows, n_cols = 4, 4
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten() if n_vars > 1 else [axes]
        
        print(f"Plotting {n_vars} refined variables in a {n_rows}x{n_cols} grid...")
        
        for idx, var_name in enumerate(variables):
            ax = axes[idx]
            
            # Sample data for faster plotting
            data = self.refined_data[var_name]
            sampled_data = data[::sample_rate, ::sample_rate]
            
            # Plot
            im = sampled_data.plot(ax=ax, cmap=cmap, add_colorbar=True, cbar_kwargs={'shrink': 0.8})
            ax.set_title(var_name, fontsize=10, fontweight='bold')
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_vars, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        print("Grid plot created successfully.")
        return fig

    def plot_refined_data(self, variable_name: str, cmap=COLOR_PALETTES['light_to_strong_blue']):
        """
        Plots a single refined data variable using matplotlib.
        
        Args:
            variable_name (str): The name of the variable to plot.
            cmap (str): The colormap to use for plotting.
        
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if variable_name not in self.refined_data:
            raise RuntimeError(f"Variable '{variable_name}' not found in refined_data. "
                               f"Run select_and_refine_variable('{variable_name}') first.")
        
        print(f"Plotting refined variable '{variable_name}'...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = self.refined_data[variable_name].plot(
            ax=ax, cmap=cmap, add_colorbar=True
        )
        ax.set_title(variable_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        plt.tight_layout()
        print("Plot created successfully.")
        return fig
    
    def plot_plottable_data(self, variable_name: str, cmap=COLOR_PALETTES['light_to_strong_blue']): 
        """
        Plots a single plottable data variable using hvplot.
        
        Args:
            variable_name (str): The name of the variable to plot.
            cmap (str or list): The colormap to use for plotting.
        
        Returns:
            hvplot object: An interactive map visualization.
        """
        if variable_name not in self.plottable_data:
            raise RuntimeError(f"Variable '{variable_name}' not found in plottable_data. "
                               f"Run prepare_for_map_overlay('{variable_name}') first.")
        
        print(f"Creating interactive map for '{variable_name}'...")
        
        plot = self.plottable_data[variable_name].hvplot.image(
            x='x', y='y',
            geo=True,
            tiles='OSM',
            cmap=cmap,
            alpha=0.7,
            frame_width=600,
            frame_height=400,
            title=variable_name,
            colorbar=True,
            xlabel="Longitude",
            ylabel="Latitude"
        )
        
        print("Interactive map created successfully.")
        return plot

    def plot_plottable_data_grid(self, variables:list=None, cmap=COLOR_PALETTES['light_to_strong_blue'], tiles='OSM', alpha=0.7, frame_width=300, frame_height=250):
        """
        Plots all plottable data variables in a grid layout using hvplot.
        
        Args:
            cmap (str or list): The colormap to use for plotting.
            tiles (str): The map tile provider (e.g., 'OSM', 'EsriImagery').
            alpha (float): The transparency of the overlay (0.0 to 1.0).
            frame_width (int): Width of each subplot in pixels.
            frame_height (int): Height of each subplot in pixels.
        
        Returns:
            holoviews.core.layout.Layout: The grid layout of maps.
        """
        if not self.plottable_data:
            raise RuntimeError("No plottable data available. Run prepare_for_map_overlay_all_vars() first.")
        
        # Limit to first 16 variables
        if variables is None:   
            variables = list(self.plottable_data.keys())[:16]
        else: 
            variables = [var for var in variables if var in self.plottable_data][:16]
        n_vars = len(variables)
        
        # Calculate grid dimensions
        if n_vars <= 4:
            n_rows, n_cols = 2, 2
        elif n_vars <= 9:
            n_rows, n_cols = 3, 3
        else:
            n_rows, n_cols = 4, 4

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(frame_height, frame_width), sharex=True, sharey=True)
        axes = axes.flatten() if n_vars > 1 else [axes]
        
        print(f"Plotting {n_vars} refined variables in a {n_rows}x{n_cols} grid...")
       
        for idx, var_name in enumerate(variables):
            ax = axes[idx]
            
            # Sample data for faster plotting
            data = self.plottable_data[var_name]
            
            # Plot
            im = data.plot(ax=ax, cmap=cmap, add_colorbar=True, cbar_kwargs={'shrink': 0.8})
            ax.set_title(var_name, fontsize=10, fontweight='bold')
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_vars, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        print("Grid plot created successfully.")
        return fig
    
    def plot_polygons_grid(self, color="darkred", alpha=0.7, figzise=(20, 20), edge_color="black", linewidth=0.5):
        """
        Plots all polygon data in a grid layout using hvplot.
        """
        if not self.polygons_data:
            raise RuntimeError("No polygon data available. Run build_polygons_all_vars() first.")
        
        variables = list(self.polygons_data.keys())[:16]
        n_vars = len(variables)
        
        if n_vars <= 4:
            n_rows, n_cols = 2, 2
        elif n_vars <= 9:
            n_rows, n_cols = 3, 3
        else:
            n_rows, n_cols = 4, 4
        
        print(f"Creating polygon map grid for {n_vars} variables in a {n_rows}x{n_cols} layout...")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figzise, sharex=True, sharey=True)
        axes = axes.flatten() if n_vars > 1 else [axes]
        
        plots = []
        for idx, var_name in tqdm(enumerate(variables), desc="Creating polygon grid", unit="map"):
            ax = axes[idx]
            gdf = self.polygons_data[var_name]

            gdf.plot(
                ax=ax,
                color=color,
                alpha=alpha,
                edgecolor=edge_color,
                linewidth=linewidth
            )
            ax.set_title(var_name, fontsize=10, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=8)
            ax.set_ylabel('Latitude', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
        
        for idx in range(n_vars, len(axes)):
            axes[idx].axis('off') 
        
        plt.tight_layout()
        print("Polygon grid plot created successfully.")
        return fig
    
    def plot_on_map(self, variable_name,cmap: list, alpha: float = 0.1, title: str = None, tiles: str = "OSM"):
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
            title = f"'{variable_name}' on {date}"

        flood_map = self.plottable_data[variable_name].hvplot.image(
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
    
    def plot_polygon_on_map(self, variable_name: str, color: str = "darkblue", alpha: float = 0.7, 
                           tiles: str = "OSM", frame_width: int = 800, frame_height: int = 600, 
                           title: str = None):
        """
        Plots a polygon overlay on an interactive map.
        
        Args:
            variable_name (str): The name of the variable to plot.
            color (str): The color for the polygons.
            alpha (float): The transparency of the overlay (0.0 to 1.0).
            tiles (str): The map tile provider (e.g., 'OSM', 'EsriImagery').
            frame_width (int): Width of the plot in pixels.
            frame_height (int): Height of the plot in pixels.
            title (str, optional): The title for the map.
        
        Returns:
            hvplot object: An interactive map with polygon overlay.
        """
        if variable_name not in self.polygons_data:
            raise RuntimeError(f"Variable '{variable_name}' not found in polygons_data. "
                             f"Run build_polygons('{variable_name}') first.")
        
        if title is None:
            date = self.dc.time.isel(time=0).dt.strftime('%Y-%m-%d').item()
            title = f"Flood Polygons for '{variable_name}' on {date}"
        
        gdf = self.polygons_data[variable_name]
        
        if len(gdf) == 0:
            print(f"Warning: No polygons to plot for '{variable_name}'")
            return None
        
        polygon_map = gdf.hvplot.polygons(
            geo=True,
            tiles=tiles,
            color=color,
            alpha=alpha,
            frame_width=frame_width,
            frame_height=frame_height,
            title=title,
            hover_cols=[],
            legend=False
        )
        
        return polygon_map

    def save_to_file(self, filepath: str, compress: bool = True):
        """
        Saves the entire MapVisualiser state to a file.
        
        Args:
            filepath (str): Path where to save the file (without extension).
            compress (bool): Whether to use compression (recommended for large datasets).
        
        The method saves:
        - All xarray DataArrays (refined_data, plottable_data) as NetCDF
        - All GeoDataFrames (polygons_data) as GeoPackage
        - Metadata and attributes as pickle
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        base_path = filepath.replace('.mapvis', '')  # Remove extension if provided
        
        print(f"Saving MapVisualiser to {base_path}...")
        
        # Create a progress bar
        total_steps = 4 + len(self.refined_data) + len(self.plottable_data) + len(self.polygons_data)
        pbar = tqdm(total=total_steps, desc="Saving MapVisualiser", unit="step")
        
        # 1. Save metadata
        pbar.set_postfix_str("Saving metadata")
        metadata = {
            'original_crs': self.original_crs,
            'bounding_box': self.bounding_box,
            'flood_variables': self.flood_variables,
            'refined_variables': list(self.refined_data.keys()),
            'plottable_variables': list(self.plottable_data.keys()),
            'polygon_variables': list(self.polygons_data.keys()),
            'save_timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(f"{base_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        pbar.update(1)
        
        # 2. Save main datacube
        pbar.set_postfix_str("Saving datacube")
        if compress:
            encoding = {var: {'zlib': True, 'complevel': 5} for var in self.dc.data_vars}
            self.dc.to_netcdf(f"{base_path}_datacube.nc", encoding=encoding)
        else:
            self.dc.to_netcdf(f"{base_path}_datacube.nc")
        pbar.update(1)
        
        # 3. Save refined_data
        pbar.set_postfix_str("Saving refined data")
        if self.refined_data:
            refined_ds = xr.Dataset(self.refined_data)
            if compress:
                encoding = {var: {'zlib': True, 'complevel': 5} for var in refined_ds.data_vars}
                refined_ds.to_netcdf(f"{base_path}_refined.nc", encoding=encoding)
            else:
                refined_ds.to_netcdf(f"{base_path}_refined.nc")
        pbar.update(1)
        
        # 4. Save plottable_data
        pbar.set_postfix_str("Saving plottable data")
        if self.plottable_data:
            plottable_ds = xr.Dataset(self.plottable_data)
            if compress:
                encoding = {var: {'zlib': True, 'complevel': 5} for var in plottable_ds.data_vars}
                plottable_ds.to_netcdf(f"{base_path}_plottable.nc", encoding=encoding)
            else:
                plottable_ds.to_netcdf(f"{base_path}_plottable.nc")
        pbar.update(1)
        
        # 5. Save polygons_data
        if self.polygons_data:
            pbar.set_postfix_str("Saving polygons")
            for var_name, gdf in self.polygons_data.items():
                if len(gdf) > 0:
                    gdf.to_file(f"{base_path}_polygon_{var_name}.gpkg", driver='GPKG')
                pbar.update(1)
        
        pbar.close()
        
        # Create an index file
        index = {
            'base_path': base_path,
            'files': {
                'metadata': f"{base_path}_metadata.pkl",
                'datacube': f"{base_path}_datacube.nc",
                'refined': f"{base_path}_refined.nc" if self.refined_data else None,
                'plottable': f"{base_path}_plottable.nc" if self.plottable_data else None,
                'polygons': {var: f"{base_path}_polygon_{var}.gpkg" for var in self.polygons_data.keys()} if self.polygons_data else {}
            }
        }
        
        with open(f"{base_path}.mapvis", 'w') as f:
            import json
            json.dump(index, f, indent=2)
        
        print(f"✓ MapVisualiser saved successfully to {base_path}")
        print(f"  Main file: {base_path}.mapvis")
        print(f"  Total files created: {len([f for f in os.listdir(os.path.dirname(base_path) if os.path.dirname(base_path) else '.') if os.path.basename(base_path) in f])}")