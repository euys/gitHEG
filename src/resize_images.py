import argparse
import os
import glob
from PIL import Image

import sys
try:
    import numpy as np
    import pyfits as pf
    import scipy.ndimage as nd
    import pylab as pl
    import os
    import heapq
    from scipy.optimize import leastsq

except ImportError:
    print
    'Error: missing one of the libraries (numpy, pyfits, scipy, matplotlib)'
    sys.exit()
sys.path.append(os.path.dirname(os.path.abspath(path)))



def main() -> object:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    args = parser.parse_args()
    resize_images(args.in_dir, args.out_dir, args.size)


def resize_images(in_dir, out_dir, size=300):
    print(in_dir, out_dir, size)
    in_files = glob.glob(os.path.join(in_dir, "*"))
    for in_file in in_files:
        basename = os.path.basename(in_file).split(".")[0] + ".png"
        out_file = os.path.join(out_dir, basename)
        resize_image(in_file, out_file, max_size=size)


def resize_image(in_file, out_file, max_size=300):
    size = max_size, max_size
    try:
        image = Image.open(in_file)
        image.thumbnail(size, Image.ANTIALIAS)
        image.save(out_file)
    except IOError:
        print("cannot create thumbnail for '%s'" % in_file)


if __name__ == "__main__":
    main()
