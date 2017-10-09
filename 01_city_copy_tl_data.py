# python imports
from __future__ import print_function
import os, glob, sys, shutil


# The main method
def main():
    currentPath = os.getcwd()

    cityscapesPath = '../../cityscapes/data'
    destinationPath = 'data/cityscapes'

    os.chdir(cityscapesPath)

    # how to search for all ground truth
    searchFine   = os.path.join( "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )
    searchCoarse = os.path.join( "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
    filesCoarse = glob.glob( searchCoarse )
    filesCoarse.sort()

    # concatenate fine and coarse
    files = filesFine + filesCoarse
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        print( "Did not find any files" )
        return

    # a bit verbose
    print("Found {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    count = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ))
    for fname in files:
        # check if it has traffic light annotations
        with open(fname, 'r') as f:
            str = f.read()
        if str.find('traffic light')>-1:
            count += 1
            # create the output filename
            dst = os.path.join(currentPath, destinationPath, fname)
            # create dst dir
            path = os.path.dirname(dst)
            if not os.path.exists(path):
                os.makedirs(path)
            # copy json file
            shutil.copy(fname, dst)

            # image file name
            ifname = fname.replace('gtFine_polygons','leftImg8bit')
            ifname = ifname.replace('gtCoarse_polygons', 'leftImg8bit')
            ifname = ifname.replace('.json', '.png')
            ifname = ifname.replace('gtFine/','leftImg8bit/')
            ifname = ifname.replace('gtCoarse/','leftImg8bit/')
            # create the output filename
            dst = os.path.join(currentPath, destinationPath, ifname)
            # create dst dir
            path = os.path.dirname(dst)
            if not os.path.exists(path):
                os.makedirs(path)
            # copy image file
            shutil.copy(ifname, dst)

        # status
        progress += 1
        print("Progress: {:>3} %".format( float(progress) * 100. / len(files) ))
        sys.stdout.flush()
    print("Copied {} files with traffic lights".format(count))


# call the main
if __name__ == "__main__":
    main()
