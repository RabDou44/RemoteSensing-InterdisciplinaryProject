import hvplot.xarray
import pyproj
from dask.distributed import wait
from odc import stac as odc_stac


class DcLoader:
    """
    A class to simplify loading data from a STAC catalog into an ODC data cube.
    """
    def __init__(self, catalog):
        """
        Initializes the loader with a STAC catalog client.

        Args:
            catalog: An active pystac_client.Client object.
        """
        self.catalog = catalog
        self.dc = None
        self.bounding_box = None
        self.time_range = None
        self.crs = None
        self.resolution = None


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
        self.crs = crs.to_string()
        self.resolution = gfm_items[0].properties['gsd']
        
        gfm_dc = odc_stac.load(
            gfm_items,
            bbox=bounding_box,
            crs=self.crs,  # Use string representation of CRS
            resolution=self.resolution,  # Use scalar resolution
            dtype='uint8',
            chunks={"x": 1024, "y": 1024, "time": 1000}
        )

        print("Persisting data cube to Dask cluster...")
        self.dc = gfm_dc.persist()
        wait(self.dc, timeout="300s")
        print("Data loaded successfully.")
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