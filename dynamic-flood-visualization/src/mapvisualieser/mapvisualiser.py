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
from bokeh.core.property.visual import ColorSpec
from dcloader.loader import DcLoader
from typing import List

class MapVisualiser:
    """
    A main class to handle bokeh mappers (here color_scales) and apply them to mappings
    """

    def __init__(self, mappings: np.ndarray = None, color_scales: List[ColorMapper] = None):
        """
        Initializes the visualizer with an array of mappings and color scales.

        Args:
            mappings (np.ndarray): The array containing mapping information.
            color_scales (List[ColorMapper], optional): List of Bokeh color mappers.
                                                            If None, uses default color scales.
        """
        self.__mappings__ = mappings
        self.__color_scales__ = color_scales or [LinearColorMapper(palette=Blues8, low=0, high=1)]

    def add_mappings(self, dcloader: DcLoader):
        """
        Imports new mappings into the visualizer.

        Args:
            mappings (np.ndarray): The array containing new mapping information.
        """
        if self.__mappings__ is not None:
            self.__mappings__  += dcloader.dc.to_numpy()
        else:
            self.__mappings__  = dcloader.dc.to_numpy()

    def add_mappings_from_array(self, mappings: np.ndarray):
        """
        Imports new mappings into the visualizer from a numpy array.

        Args:
            mappings (np.ndarray): The array containing new mapping information.
        """
        if self.__mappings__ is not None:
            self.__mappings__  += mappings
        else:
            self.__mappings__  = mappings

    def apply_color_scale_to_mapping(self, mapping_index=0, color_scale_index=0, downsample_factor=4):
        """
        Plots the flood likelihood for a specific time index.

        Args:
            time_index (int): The time index to visualize.
            downsample_factor (int): Factor by which to downsample the data for visualization.

        Returns:
            hvplot object: The hvplot image of the flood likelihood.
        """
        self.__mappings__[mapping_index][:,::downsample_factor,::downsample_factor].hvplot.image(cmap=self.__color_scales__[color_scale_index],
                                                         x='x', y='y', rasterize=True,
                                                         width=800, height=400)
        