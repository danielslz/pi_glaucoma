from math import pi
from skimage.measure import label, regionprops


def get_ellipse_area(ellipse):
    region = label(ellipse)
    props = regionprops(region)
    props = props[0]

    major_radius = props.major_axis_length/2
    minor_radius = props.minor_axis_length/2

    return pi*major_radius*minor_radius


def get_cdr(cup, disc):
    cup_area = get_ellipse_area(cup)
    disc_area = get_ellipse_area(disc)

    return cup_area/disc_area
