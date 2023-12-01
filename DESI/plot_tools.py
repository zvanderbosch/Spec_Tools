import colorsys
import matplotlib.colors as mc

"""
Tools for plotting.

Author: Zach Vanderbosch (Caltech)
Updated: 2023 June 28
"""


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) 
    by the given amount. Input can be matplotlib color 
    string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    
    try:
        c = mc.cnames[color]
    except:
        c = color
        
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    rgb = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    hexcol = mc.to_hex(rgb)
    
    return hexcol