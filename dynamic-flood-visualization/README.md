# Dynamic Flood Visualization

A Python package for visualizing flood data from STAC catalogs using ODC (Open Data Cube) and interactive plotting with hvplot.

## Overview

This package provides tools to load and visualize Global Flood Monitoring (GFM) data from STAC catalogs. It simplifies the process of downloading satellite data and creating interactive visualizations for flood analysis.

## Features

- 🌍 Load Global Flood Monitoring data from STAC catalogs
- 📊 Interactive visualization with hvplot
- ⚡ Dask-powered distributed computing for large datasets
- 🗺️ Automatic CRS and resolution detection
- 📈 Built-in visualization methods for flood extent analysis

## Installation

### Prerequisites

- Python 3.9 or higher
- Git (for cloning the repository)

### Option 1: Development Installation (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RabDou44/RemoteSensing-InterdisciplinaryProject.git
   cd RemoteSensing-InterdisciplinaryProject/dynamic-flood-visualization
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```

5. **Install Jupyter dependencies for notebook support:**
   ```bash
   pip install jupyter jupyter-bokeh
   ```

### Option 2: Direct Installation from Source

```bash
pip install git+https://github.com/RabDou44/RemoteSensing-InterdisciplinaryProject.git#subdirectory=dynamic-flood-visualization
```

## Dependencies

The package automatically installs the following dependencies:

- `odc-stac` - Open Data Cube STAC integration
- `pystac-client` - STAC client for catalog access
- `hvplot` - Interactive plotting
- `dask[distributed]` - Distributed computing
- `pyproj` - Coordinate system transformations
- `xarray` - N-dimensional labeled arrays

## Quick Start

### Basic Usage

```python
import pystac_client
from dask.distributed import Client
from dcloader import DcLoader

# Set up Dask client
client = Client(processes=False, threads_per_worker=2, n_workers=3, memory_limit="12GB")

# Connect to EODC STAC catalog
eodc_catalog = pystac_client.Client.open("https://stac.eodc.eu/api/v1/")

# Define area of interest and time range
time_range = '2022-09-23/2022-09-24'
bounding_box = [67.9, 27.0, 68.7, 27.8]  # [min_lon, min_lat, max_lon, max_lat]

# Load flood data
dc_loader = DcLoader(eodc_catalog)
flood_data = dc_loader.load_GFM_data(time_range, bounding_box)

# Visualize flood extent
flood_data['tuw_flood_extent'].hvplot.image(x="x", y="y", title="Flood Extent")
```

### Using the Built-in Visualization Method

```python
# Alternative: Use the built-in visualization method
visualization = dc_loader.visualize_band('water')
visualization.show()
```

## Available Data Bands

The Global Flood Monitoring data typically includes the following bands:

- `tuw_flood_extent` - Flood extent classification
- `water` - Water detection
- `flood` - Flood classification
- And others depending on the specific dataset

To see available bands in your loaded data:
```python
print(list(flood_data.data_vars))
```

## Example Notebook

Check out the `test/test.ipynb` notebook for a complete example of loading and visualizing flood data for Pakistan.

## Project Structure

```
dynamic-flood-visualization/
├── src/
│   ├── dcloader/
│   │   ├── __init__.py
│   │   └── loader.py          # Main DcLoader class
│   └── visualise/             # Future visualization modules
├── test/
│   └── test.ipynb            # Example usage notebook
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Development

### Setting up Development Environment

1. Clone the repository and navigate to the project directory
2. Create and activate a virtual environment
3. Install in editable mode: `pip install -e .`
4. Install development dependencies: `pip install jupyter jupyter-bokeh`

### Running Tests

Currently, testing is done through the example notebook in `test/test.ipynb`.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'jupyter_bokeh'**
   ```bash
   pip install jupyter-bokeh
   ```

2. **Dask cluster connection issues**
   - Ensure sufficient memory is allocated
   - Check firewall settings for local cluster

3. **STAC catalog connection timeout**
   - Check internet connection
   - Verify catalog URL is accessible

### Memory Requirements

- Minimum: 8GB RAM
- Recommended: 16GB+ RAM for larger datasets
- Adjust Dask memory limits based on your system

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the example notebook
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- EODC for providing STAC catalog access
- Open Data Cube community
- HoloViews/hvplot for interactive visualization tools

## Links

- [GitHub Repository](https://github.com/RabDou44/RemoteSensing-InterdisciplinaryProject)
- [Issues](https://github.com/RabDou44/RemoteSensing-InterdisciplinaryProject/issues)

## What do we show the end user
- One version with polygons for extend
- one raw version (baseline)
- one raw version with map (baseline)
- Polygons with extend and likelihood?
- Polygons with extend and likelihood with binning?
Optional: Only satellite with water visible?

Feedback Astrid, Thais: Around the 20-22th

Jonas: Survey setup, google forms
Jakob: Discusses with Adam the polygons 
Adam: Polygons with binning and scales

