from movie import process_movie
from detect import pipeline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main():
    import sys
    images = sys.argv[1:]

    for fname in images:
        image = mpimg.imread(fname)
        out = pipeline(image)

        mpimg.imsave(fname+'-output.jpg', out)


if __name__ == '__main__':
    main()
