import numpy as np

def arrange_in_grid(array, ncols=2):
    """
    from https://stackoverflow.com/questions/42040747/more-idomatic-way-to-display-images-in-a-grid-with-numpy
    arranges images in a grid
    """
    array = np.asarray(array)
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result
