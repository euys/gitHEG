import argparse
import os
import fnmatch
import glob
import shutil

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="directory of images")
    parser.add_argument("--anno_dir", help="directory of annotations")
    parser.add_argument("--dest_dir", help="directory for destination")
    args = parser.parse_args()
    leave_unused(args.image_dir, args.anno_dir, args.dest_dir)


def leave_unused(image_dir, anno_dir, dest_dir):
    print(image_dir)
    print(anno_dir)
    print(dest_dir)

    annotated = get_filenames(anno_dir, "*.xml")
    image_paths = glob.glob(os.path.join(image_dir, "*"))
    no_anno = 0
    for image_path in image_paths:
        if is_annotated(image_path, annotated):
            shutil.copy2(image_path, dest_dir)
        else:
            print("no anno", image_path)
            no_anno += 1
    print("{} removed out of {}".format(no_anno, len(image_paths)))


def get_filenames(path, pattern):
    pattern = "*.xml"
    names = [fn.split(".")[0] for fn in
             fnmatch.filter(os.listdir(path), pattern)]

    return names


def is_annotated(image_path, annotated_names):
    basename = os.path.basename(image_path)
    image_filename = basename.split(".")[0]
    return image_filename in annotated_names


if __name__ == "__main__":
    main()
