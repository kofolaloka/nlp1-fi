import argparse
import os
import os.path
import sys
import glob

def argsdirs(title):
    ap = argparse.ArgumentParser(title)
    ap.add_argument("-i", nargs=1, help="directory containing input data")
    ap.add_argument("-o", nargs=1, help="directory containing output data")
    ap.add_argument("-m", nargs=1, help="subset of v,s,o ; comma separated")
    args = ap.parse_args()
    try:
        i = args.i[0]
        o = args.o[0]
    except:
        ap.print_help()
        sys.exit(-1)

    print "input directory:%s"%i
    print "output directory:%s"%o

    if not os.path.exists(o):
        try:
            os.mkdir(o)
        except Exception as e:
            raise Exception("cannot create directory %s: %s"%(o,str(e)))

    try:
        m = args.m[0].split(',')
    except:
        m = ['v','s','o']
    
    for curr in [i,o]:
        if not os.path.exists(curr):
            raise Exception("directory %s does not exist"%curr)

    return i,o,m

def filenames(dirname):
    return glob.glob(os.path.join(dirname,"*"))

def new_filename(dirname, prev):
    ret =  os.path.join(dirname,os.path.basename(prev))
    assert ret != prev,"new filename %s == previous filename %s"%(ret,prev)
    return ret
