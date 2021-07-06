import numpy as np

from skimage import img_as_ubyte, exposure
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.util.shape import view_as_windows


LBP = 'lbp'
HARALICK = 'haralick'
COLOR_MOMENTS = 'color_moments'


def get_lbp_histograms(img, radius):
    eps = 1e-7
    n_points = 8 * radius
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    (hist, _) = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return hist.tolist()


def describe_lbp(img, radius=1):
    # rgb to gray
    img = rgb2gray(img)

    # contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    # split image in 16 blocks
    M = img.shape[0]//3
    N = img.shape[1]//3
    blocks = [img[x:x+M, y:y+N] for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], N)]

    histograms = []
    # compute lbp for each block
    for block in blocks:
        histograms += get_lbp_histograms(block, radius)
    
    # append zeros to make list at same size
    total = ((radius * 8) + 2) * 16
    histograms.extend([0.0] * (total - len(histograms)))

    return histograms


def describe_haralick(img):
    # rgb to gray
    img = rgb2gray(img)
    
    # contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    
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
