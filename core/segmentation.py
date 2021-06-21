import cv2
import numpy as np

from skimage.draw import ellipse_perimeter
from skimage.measure import label, regionprops
from skimage.transform import hough_ellipse


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


def ellipse_fitting_3(img):
    output = np.zeros_like(img)

    ((centx, centy), (width, height), angle) = cv2.fitEllipse(img)

    cv2.ellipse(output, (int(centx),int(centy)), (int(width/2),int(height/2)), angle, 0, 360, (0,0,255), 2)

    # output[cy, cx] = 1
    return output


def ellipse_fitting_2(img):
    output = np.zeros_like(img)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    # result = hough_ellipse(img, accuracy=20, threshold=250, min_size=100, max_size=120)
    result = hough_ellipse(img)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    output[cy, cx] = 1
    return output


def ellipse_fitting(img):
    output = np.zeros_like(img)#.astype(int)

    region = label(img)
    props = regionprops(region)
    props = props[0]
    # row, col  = props.centroid
    # rr, cc = ellipse_perimeter(int(row), int(col), int(props.minor_axis_length*0.5), int(props.major_axis_length*0.5), orientation=props.orientation, shape=img.shape)
    # output[rr, cc] = 1

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


# def draw_ellipse_fitting(img):
#     alpha = 5
#     beta = 3
#     N = 500
#     DIM = 2

#     np.random.seed(2)

#     # Generate random points on the unit circle by sampling uniform angles
#     theta = np.random.uniform(0, 2*np.pi, (N,1))
#     eps_noise = 0.2 * np.random.normal(size=[N,1])
#     circle = np.hstack([np.cos(theta), np.sin(theta)])

#     # Stretch and rotate circle to an ellipse with random linear tranformation
#     B = np.random.randint(-3, 3, (DIM, DIM))
#     noisy_ellipse = circle.dot(B) + eps_noise

#     # Extract x coords and y coords of the ellipse as column vectors
#     X = noisy_ellipse[:,0:1]
#     Y = noisy_ellipse[:,1:]

#     # Formulate and solve the least squares problem ||Ax - b ||^2
#     A = np.hstack([X**2, X * Y, Y**2, X, Y])
#     b = np.ones_like(X)
#     x = np.linalg.lstsq(A, b)[0].squeeze()

#     # Print the equation of the ellipse in standard form
#     print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1],x[2],x[3],x[4]))

#     # Plot the noisy data
#     plt.scatter(X, Y, label='Data Points')

#     # Plot the original ellipse from which the data was generated
#     phi = np.linspace(0, 2*np.pi, 1000).reshape((1000,1))
#     c = np.hstack([np.cos(phi), np.sin(phi)])
#     ground_truth_ellipse = c.dot(B)
#     plt.plot(ground_truth_ellipse[:,0], ground_truth_ellipse[:,1], 'k--', label='Generating Ellipse')

#     # Plot the least squares ellipse
#     x_coord = np.linspace(-5,5,300)
#     y_coord = np.linspace(-5,5,300)
#     X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
#     Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
#     plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

#     plt.legend()
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.show()