import os
import hvplot.xarray
import pyproj
from dask.distributed import wait
from odc import stac as odc_stac
import xarray as xr
import numpy as np


class DcLoader:
    """
    A class to simplify loading data from a STAC catalog into an ODC data cube.
    """
    def __init__(self, catalog, netcdf_path=None):
        """
        Initializes the loader with a STAC catalog client.

        Args:
            catalog: An active pystac_client.Client object.
        """
        if netcdf_path is not None:
            # Load from NetCDF file
            self._load_from_netcdf_constructor(netcdf_path)
        else:
            self.catalog = catalog
            self.dc = None
            self.bounding_box = None
            self.time_range = None
            self.crs = None
            self.resolution = None

    def _load_from_netcdf_constructor(self, netcdf_path):
        """
        Internal method to load datacube and attributes from NetCDF file.
        
        Args:
            netcdf_path (str): Path to the NetCDF file.
        """
        if not os.path.exists(netcdf_path):
            raise FileNotFoundError(f"NetCDF file not found: {netcdf_path}")
        
        print(f"Loading data from {netcdf_path}...")
        self.dc = xr.open_dataset(netcdf_path)
        
        # Reconstruct attributes from NetCDF metadata
        self.catalog = None  # Catalog connection not preserved
        
        # Try to load saved metadata
        if 'bounding_box' in self.dc.attrs:
            self.bounding_box = self.dc.attrs['bounding_box']
        else:
            self.bounding_box = None
            print("Warning: bounding_box not found in file metadata")
        
        if 'time_range' in self.dc.attrs:
            self.time_range = self.dc.attrs['time_range']
        else:
            self.time_range = None
            print("Warning: time_range not found in file metadata")
        
        if 'crs' in self.dc.attrs:
            self.crs = self.dc.attrs['crs']
        else:
            # Try to extract from spatial_ref coordinate
            if 'spatial_ref' in self.dc.coords:
                try:
                    self.crs = self.dc.coords['spatial_ref'].attrs.get('crs_wkt', None)
                except:
                    self.crs = None
            else:
                self.crs = None
            if self.crs is None:
                print("Warning: CRS not found in file metadata")
        
        if 'resolution' in self.dc.attrs:
            self.resolution = self.dc.attrs['resolution']
        else:
            self.resolution = None
            print("Warning: resolution not found in file metadata")
        
        print(f"Data loaded successfully from {netcdf_path}")
        print(f"  - Bounding box: {self.bounding_box}")
        print(f"  - Time range: {self.time_range}")
        print(f"  - CRS: {self.crs}")
        print(f"  - Resolution: {self.resolution}")

    def _add_metadata_to_dataset(self):
        """
        Adds DcLoader metadata to the xarray Dataset attributes.
        This allows the metadata to be saved with the NetCDF file.
        """
        if self.dc is not None:
            if self.bounding_box is not None:
                self.dc.attrs['bounding_box'] = self.bounding_box
            if self.time_range is not None:
                self.dc.attrs['time_range'] = self.time_range
            if self.crs is not None:
                self.dc.attrs['crs'] = self.crs.to_string()
            if self.resolution is not None:
                self.dc.attrs['resolution'] = self.resolution

    def load_GFM_data(self, time_range, bounding_box):
        """
        Searches for and loads Global Flood Monitoring (GFM) data.

        Args:
            time_range (str): The time range for the data search (e.g., 'YYYY-MM-DD/YYYY-MM-DD').
            bounding_box (list): The bounding box [min_lon, min_lat, max_lon, max_lat].

        Returns:
            xarray.Dataset: The loaded data cube, persisted in Dask memory.
        """
        self.bounding_box = bounding_box
        self.time_range = time_range
        search = self.catalog.search(collections="GFM", bbox=bounding_box, datetime=time_range)
        gfm_items = search.item_collection()

        if not gfm_items:
            raise ValueError("No GFM items found for the given time range and bounding box.")

        crs = pyproj.CRS.from_wkt(gfm_items[0].properties["proj:wkt2"])
        self.resolution = gfm_items[0].properties['gsd']
        self.crs = crs
        gfm_dc = odc_stac.load(
            gfm_items,
            bbox=bounding_box,
            crs=self.crs.to_string(),  # Use string representation of CRS
            resolution=self.resolution,  # Use scalar resolution
            dtype='uint8',
            chunks={"x": 1024, "y": 1024, "time": 1000}
        )

        print("Persisting data cube to Dask cluster...")
        self.dc = gfm_dc.persist()
        wait(self.dc, timeout="300s")
        print("Data loaded successfully.")

        self._add_metadata_to_dataset()

        return self.dc

    def visualize_band(self, band_name):
        """
        Generates an interactive plot for a specific band using hvplot.

        Args:
            band_name (str): The name of the band to visualize (e.g., 'water').

        Returns:
            hvplot object: An interactive plot that can be displayed in a notebook.
        """
        if self.dc is None:
            raise RuntimeError("Data has not been loaded yet. Call load_GFM_data() first.")
        
        if band_name in self.dc:
            return self.dc[band_name].hvplot.image(x="x", y="y", cmap="Blues", title=band_name)
        else:
            available_bands = list(self.dc.data_vars)
            raise ValueError(f"Band '{band_name}' not found. Available bands are: {available_bands}")

    def get_metadata(self):
        """
        Returns a dictionary of all DcLoader metadata.
        
        Returns:
            dict: Dictionary containing all metadata attributes.
        """
        return {
            'bounding_box': self.bounding_box,
            'time_range': self.time_range,
            'crs': self.crs,
            'resolution': self.resolution,
            'has_data': self.dc is not None,
            'data_vars': list(self.dc.data_vars) if self.dc is not None else None,
            'dimensions': dict(self.dc.dims) if self.dc is not None else None
        }

    def save_to_netcdf(self, output_path):
        """
        Saves the current data cube to a NetCDF file.

        Args:
            output_path (str): The file path to save the NetCDF file.
        """
        if self.dc is None:
            raise RuntimeError("Data has not been loaded yet. Call load_GFM_data() first.")

        # Ensure metadata is added before saving
        self._add_metadata_to_dataset()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print("Computing data before saving...")
        dc_computed = self.dc.compute()
        dc_computed.to_netcdf(output_path, engine="netcdf4")
        print(f"Data cube saved to {output_path}")


    def load_from_netcdf(self, input_path, lazy=False):
        """
        Loads a data cube from a NetCDF file.

        Args:
            input_path (str): The file path of the NetCDF file to load.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"NetCDF file not found: {input_path}")
        
        print(f"Loading data from {input_path}...")
        
        if lazy:
            self.dc = xr.open_dataset(input_path, chunks='auto')
            print(f"Data loaded lazily (requires Dask client)")
        else:
            self.dc = xr.open_dataset(input_path)
            print(f"Data loaded into memory")
        
        # Reconstruct attributes from saved metadata
        if 'bounding_box' in self.dc.attrs:
            self.bounding_box = self.dc.attrs['bounding_box']
        
        if 'time_range' in self.dc.attrs:
            self.time_range = self.dc.attrs['time_range']
        
        if 'crs' in self.dc.attrs:
            self.crs = pyproj.CRS.from_string(self.dc.attrs['crs'])
        elif 'spatial_ref' in self.dc.coords:
            try:
                self.crs = self.dc.coords['spatial_ref'].attrs.get('crs_wkt', None)
            except:
                self.crs = None
        
        if 'resolution' in self.dc.attrs:
            self.resolution = self.dc.attrs['resolution']
        
        print(f"Reconstructed attributes:")
        print(f"  - Bounding box: {self.bounding_box}")
        print(f"  - Time range: {self.time_range}")
        print(f"  - CRS: {self.crs}")
        print(f"  - Resolution: {self.resolution}")
        
        return self.dc
    
    def compute_swath_mosaic(
        self,
        flood_variables=None,
        fill_value=255,
        mask_var="reference_water_mask",
        water_mask_value=1,
        persist=True,
        debug=False,
        mask_variables=None,
    ):
        """
        Creates a temporal swath mosaic for the supplied flood variables and updates the datacube.

        Args:
            flood_variables (list[str] | None): Variables to mosaic; defaults to vars containing
                'flood' or 'likelihood' in their names.
            fill_value (int): Value that marks gaps / invalid pixels.
            mask_var (str): Name of the static water mask variable.
            water_mask_value (int | float | None): Value within the mask to treat as water (set to
                None to skip masking).
            persist (bool): Whether to persist the updated Dataset on the connected Dask cluster.
            mask_variables (list[str] | None): Mask-like variables that should also be mosaicked
                (defaults to common mask layers if available).

        Returns:
            xarray.Dataset: Updated dataset with mosaicked flood variables.
        """
        if self.dc is None:
            raise RuntimeError("Data has not been loaded yet. Call load_GFM_data() first.")

        if flood_variables is None:
            flood_variables = [
                var
                for var in self.dc.data_vars
                if any(keyword in var.lower() for keyword in ("flood", "likelihood"))
            ]

        if not flood_variables:
            raise ValueError("No flood variables provided or detected for swath mosaicking.")

        mask_variables = list(mask_variables or [])
        if mask_var and mask_var not in mask_variables:
            mask_variables.append(mask_var)
        mask_variables = [var for var in mask_variables if var in self.dc.data_vars]

        target_variables = []
        for candidate in flood_variables + [v for v in mask_variables if v not in flood_variables]:
            if candidate not in target_variables:
                target_variables.append(candidate)

        static_mask = None
        if water_mask_value is not None and mask_var in self.dc.data_vars:
            mask_da = self.dc[mask_var]
            if "time" in mask_da.dims:
                mask_da = mask_da.isel(time=0)
            static_mask = (mask_da == water_mask_value).load()

        updated_vars = []
        for var in target_variables:
            if var not in self.dc.data_vars:
                print(f"Skipping '{var}' (not present in dataset).")
                continue

            data_array = self.dc[var]
            if "time" not in data_array.dims:
                print(f"Skipping '{var}' (no time dimension).")
                continue

            var_fill_value = data_array.attrs.get("_FillValue", fill_value)
            if var_fill_value is None:
                var_fill_value = fill_value

            is_mask_variable = var in mask_variables
            working_da = data_array.astype(np.int32)
            working_da.name = var
            if static_mask is not None and not is_mask_variable:
                working_da = working_da.where(~static_mask, other=var_fill_value)

            mosaic = self._temporal_swath_mosaic(working_da, fill_value=var_fill_value, debug=debug)
            mosaic = mosaic.astype(data_array.dtype)

            repeated_slices = [mosaic.copy(deep=True) for _ in range(data_array.sizes["time"])]
            repeated = xr.concat(repeated_slices, dim="time")
            repeated = repeated.assign_coords(time=data_array.coords["time"])
            repeated = repeated.transpose(*data_array.dims)
            repeated.attrs = data_array.attrs

            data_chunks = getattr(data_array, "chunks", None)
            if data_chunks is not None:
                chunk_map = {dim: chunks for dim, chunks in zip(data_array.dims, data_chunks)}
                repeated = repeated.chunk(chunk_map)

            self.dc[var] = repeated
            updated_vars.append(var)

        if persist:
            self.dc = self.dc.persist()

        print(f"Swath mosaic applied to {len(updated_vars)} variable(s).")
        return self.dc

    @staticmethod
    def _temporal_swath_mosaic(data_band: xr.DataArray, fill_value: int | float, debug: bool = False) -> xr.DataArray:
        """
        Performs first-valid-pixel compositing over time for a single variable.
        """
        if data_band.sizes.get("time", 0) == 0:
            raise ValueError("No data found in the time series for mosaicking.")

        base_slice = data_band.isel(time=0).copy(deep=True).load()
        base_array = base_slice.values
        missing_mask = (base_array == fill_value) | np.isnan(base_array)

        for t_index in range(1, data_band.sizes["time"]):
            if not missing_mask.any():
                break
            fill_slice = data_band.isel(time=t_index).load()
            fill_array = fill_slice.values
            valid_fill = (fill_array != fill_value) & (~np.isnan(fill_array))
            fill_candidates = missing_mask & valid_fill
            pixels_filled = np.count_nonzero(fill_candidates)
            if pixels_filled:
                if debug:
                    print(
                        f"[Swath Mosaic] var='{getattr(data_band, 'name', 'unknown')}' "
                        f"time_idx={t_index} filled {pixels_filled} pixels using swath from time index {t_index}."
                    )
                base_array[fill_candidates] = fill_array[fill_candidates]
                missing_mask[fill_candidates] = False

        base_slice.values = base_array
        return base_slice

