"""
Custom linear color schemes for flood data visualization
"""



COLOR_PALETTES = {
    # Blue-to-red gradient
    'blue_to_red': ['#08306b', '#08519c', '#4292c6', '#9ecae1', '#ffffcc', '#fed976', '#fd8d3c', '#e31a1c'],
    
    # Green-to-brown gradient
    'green_to_brown': ['#006837', '#31a354', '#78c679', '#c2e699', '#ffffcc', '#fec44f', '#fe9929', '#d95f0e', '#993404'],
    
    # White-to-blue monochromatic scheme
    'white_to_blue': ['#ffffff', '#f0f9ff', '#bae6fd', '#7dd3fc', '#38bdf8', '#0ea5e9', '#0284c7', '#075985'],
    
    # Viridis-inspired scheme
    'purple_to_yellow': ['#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825'],
    
    # Diverging red-blue scheme
    'red_to_blue_diverging': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'],
    
    # Single black color
    'solid_black': ['#000000'],
}

def get_color_scheme(scheme_name):
    """
    Parse color scheme name and return the corresponding color palette.
    
    Args:
        scheme_name (str): Exact name of the color scheme.
    
    Returns:
        list: Color palette as list of hex colors
        
    Raises:
        ValueError: If scheme name is not recognized
    """
    if scheme_name not in COLOR_PALETTES:
        available_schemes = list(COLOR_PALETTES.keys())
        raise ValueError(f"Unknown color scheme: '{scheme_name}'. Available schemes: {available_schemes}")
    
    return COLOR_PALETTES[scheme_name]

def list_available_schemes():
    """
    List all available color schemes.
    
    Returns:
        list: List of available color scheme names
    """
    return list(COLOR_PALETTES.keys())