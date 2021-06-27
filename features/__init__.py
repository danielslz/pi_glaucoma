import numpy as np

from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops


LBP = 'lbp'
HARALICK = 'haralick'
COLOR_MOMENTS = 'color_moments'


def describe_lbp(img, radius=12):
    eps = 1e-7
    n_points = 8 * radius
    img = rgb2gray(img)
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return hist.tolist()


def describe_haralick(img):
    img = rgb2gray(img)
    img = img_as_ubyte(img)
    img = np.asarray(img, dtype='int32')
    angles = [0, np.pi/4, np.pi/2, (3 * np.pi)/4]

    output = []
    for theta in angles:
        glcm = greycomatrix(img, [1], [theta], levels=img.max()+1, symmetric=False, normed=True)
        output.append(greycoprops(glcm, 'contrast')[0][0])
        output.append(greycoprops(glcm, 'energy')[0][0])
        output.append(greycoprops(glcm, 'homogeneity')[0][0])
        output.append(greycoprops(glcm, 'correlation')[0][0])
        output.append(greycoprops(glcm, 'dissimilarity')[0][0])
        output.append(greycoprops(glcm, 'ASM')[0][0])

    return output


def describe_color_moments(img):
    img = np.asarray(img)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
                
    meanR = np.mean(R)
    meanG = np.mean(G)
    meanB = np.mean(B)

    varianceR = np.var(R)
    varianceG = np.var(G)
    varianceB = np.var(B)
    
    differenceR = 0.0
    differenceG = 0.0
    differenceB = 0.0

    for i in range (len(img)):
        for j in range (len(img[0])):
            differenceR = differenceR - np.float_power((R[i][j] - meanR), 3)
            differenceG = differenceG - np.float_power((G[i][j] - meanG), 3)
            differenceB = differenceB - np.float_power((B[i][j] - meanB), 3)

    N = len(img) * len(img[0])

    skewnessR = np.float_power((np.abs(differenceR)/N), 1/3.)
    skewnessG = np.float_power((np.abs(differenceG)/N), 1/3.)
    skewnessB = np.float_power((np.abs(differenceB)/N), 1/3.)

    return [meanR, meanG, meanB, varianceR, varianceG, varianceB, skewnessR, skewnessG, skewnessB]
