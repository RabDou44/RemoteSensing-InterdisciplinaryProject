# RemoteSensing-InterdisciplinaryProject: Techniques for Dynamic Visualization of Time-Varying Flood Data
Plan:
 - [ ] proposal of scientific question
 - [ ] preparing proposal + basic scratch of framework 

# Materials - links
- [python-dask](https://docs.eodc.eu/services/dask.html)
- https://github.com/thaisbeham/Flood_visualization
- metadata https://services.eodc.eu/browser/#/v1/collections/GFM?.language=en


## Expose
[Project Report (PDF)](dynamic-flood-visualization/docs/expose.pdf)

## Framwork design:
One main class which stores all ndarray (class Mapping) and preproccess it,transform, modify it.
It stores also differe color scales (custom one) which when called with visualise method prints it with hvplot

```
Class MapVisualiser:
    __mappings__ = np.ndarray 
    __color_scales__ = { "name of the cmap" :linear_cmap() }

    def visualise(self, index_cmap, index_of_mapping):
        pass

    def preprossing(self, index_of_mapping):
        pass
    
    def smoth_mapping(self, index_of_mapping):
        pass

```
 ## Plan (1 - 2 weeks):
| Task | Who? | Comment |
| ---- |------| --------|
| create framework for work (MapVisualiser, DcLoader)| Adam |  [mappers by bokeh](https://docs.bokeh.org/en/dev-3.0/docs/user_guide/styling/palettes_mappers.html) | 
|  test some color scales with different split for values of likelihookd, extent or likelihood*extent  | everyone |
| effectiveness measure | Jonas | testing combinations of output measure of algorithms (ensemble, dlr, list, tuw)|
| smoothing regions | Adam, Jakob | use some polygons  |

