import numpy as np
import math

from skimage.draw import ellipse
from skimage.measure import label, regionprops


def get_neighborhood(x, y, shape):
    out = []
    max_x = shape[1] - 1
    max_y = shape[0] - 1
    
    # top left
    out_x = min(max(x - 1, 0), max_x)
    out_y = min(max(y - 1, 0), max_y)
    out.append((out_x, out_y))
    
    # top center
    out_x = x
    out_y = min(max(y - 1, 0), max_y)
    out.append((out_x, out_y))
    
    # top right
    out_x = min(max(x + 1, 0), max_x)
    out_y = min(max(y - 1, 0), max_y)
    out.append((out_x, out_y))
    
    # left
    out_x = min(max(x - 1, 0), max_x)
    out_y = y
    out.append((out_x, out_y))
    
    # right
    out_x = min(max(x + 1, 0), max_x)
    out_y = y
    out.append((out_x, out_y))
    
    # bottom left
    out_x = min(max(x - 1, 0), max_x)
    out_y = min(max(y + 1, 0), max_y)
    out.append((out_x, out_y))
    
    # bottom center
    out_x = x
    out_y = min(max(y + 1, 0), max_y)
    out.append((out_x, out_y))
    
    # bottom right
    out_x = min(max(x + 1, 0), max_x)
    out_y = min(max(y + 1, 0), max_y)
    out.append((out_x, out_y))
    
    return out


def region_growing(img, start, threshold=10):
    output = np.zeros_like(img)
    
    seeds = [start]
    visited = {}

    while(len(seeds) > 0):
        seed = seeds[0]
        visited[seed] = True
        seed_intensity = int(img[start])
        output[seed] = 255
        nb = get_neighborhood(seed[0], seed[1], img.shape)
        for xy in nb:
            try:
                xy_intensity = int(img[xy])
                diff = abs(seed_intensity - xy_intensity)
                if diff < threshold:
                    output[xy] = 255
                    try:
                        if visited[xy]:
                            pass
                    except KeyError:
                        seeds.append(xy)
                    visited[xy] = True
            except IndexError:
                pass
            else:
                output[xy] = 0
        seeds.pop(0)

    return output


def ellipse_fitting(img):
    output = np.zeros_like(img)

    region = label(img)
    props = regionprops(region)
    props = props[0]

    yc, xc = [int(round(x)) for x in props.centroid]
    orientation = props.orientation
    major_axis = int(round(props.major_axis_length/2.0))
    minor_axis = int(round(props.minor_axis_length/2.0))
    rotation =  orientation + (math.pi/2.0)
    cy, cx = ellipse(yc, xc, minor_axis, major_axis, rotation=rotation)
    try:
        output[cy, cx] = 1
    except IndexError:
        y_max = img.shape[0] - 1
        x_max = img.shape[1] - 1
        cy[cy > y_max] = (y_max - 1) 
        cx[cx > x_max] = (x_max - 1)
        output[cy, cx] = 1

    return output


def draw_ellipse_fitting(img, ellipse):
    output = img.copy()

    output[ellipse > 0] = (0, 0, 255)

    return output
