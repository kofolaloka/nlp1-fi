import pandas as pd
import argparse
import os
import glob

# local imports
import triple

def main():
    ap = argparse.ArgumentParser("Counting the triples")
    ap.add_argument("-d", nargs=1, help="directory containing the preprocessed data")
    ap.add_argument("-c", nargs=1, help="directory containing the counted triples")

    ap.add_argument("-m", nargs=1, help="members to group and collapse in the sum, comma separated")
    args = ap.parse_args()
    d = args.d[0]
    c = args.c[0]
    try:
        m = args.m[0].split(',')
    except:
        m = ['v','s','o']
    for curr in [c,d]:
        if not os.path.exists(curr):
            raise Exception("directory %s does not exist"%curr)
    
    for filename in glob.glob(os.path.join(d,"*")):
        print "processing file %s.."%filename
        tome_in = triple.Tome(filename)
        filename_out = os.path.join(c,os.path.basename(filename))
        print "writing to %s.."%filename_out
        tome_out = triple.Tome(filename_out)
        
        writer = tome_out.writer()
        for tr in tome_in.group_sum(m):
            writer(tr)

if __name__=="__main__":
    main()

