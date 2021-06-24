import cv2
import numpy as np

from skimage.draw import ellipse_perimeter
from skimage.feature import canny
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
    major_axis = int(round(props.major_axis_length/2.))
    minor_axis = int(round(props.minor_axis_length/2.))
    cy, cx = ellipse_perimeter(yc, xc, minor_axis, major_axis, orientation)
    output[cy, cx] = 1

    return output


def draw_ellipse_fitting(img, ellipse):
    output = img.copy()

    output[ellipse > 0] = (0, 0, 255)

    return output


def ellipse_fitting_2(img):
    from ellipse import LsqEllipse

    output = np.zeros_like(img)

    noisy_ellipse = canny(img)

    result = np.where(noisy_ellipse == True)
    points = np.array(list(zip(result[0], result[1])))

    # reg = LsqEllipse().fit(points)
    # center, width, height, phi = reg.as_parameters()

    # yc, xc = [int(round(x)) for x in center]
    # orientation = -1*phi
    # major_axis = int(round(max(width, height)))
    # minor_axis = int(round(min(width, height)))
    
    (xc, yc), (w, h), theta = cv2.fitEllipse(points)
    xc = int(round(xc))
    yc = int(round(yc))
    orientation = -1*theta
    major_axis = int(round(max(w, h)/2.))
    minor_axis = int(round(min(w, h)/2.))

    cy, cx = ellipse_perimeter(xc, yc, minor_axis, major_axis, orientation=orientation, shape=img.shape)
    output[cy, cx] = 1

    # output = canny(img).astype(np.uint8)
    # cv2.ellipse(output, (xc, yc), (int(w/2), int(h/2)), theta, 0, 360, (0,0,255), 1)

    return output