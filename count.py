import pandas as pd
import argparse
import os
import gzip
import glob

# local imports
import triple

def process(filename):
    handle = gzip.open(filename, 'rb')
    df = pd.DataFrame.from_csv(
        handle,
        sep='\t',
        #names=triple.Triple.members(),
        header=None,
        index_col=None
    )
    counted = df.groupby([0,1,2]).sum().reset_index()
    return counted
    
def main():
    ap = argparse.ArgumentParser("Counting the triples")
    ap.add_argument("-d", nargs=1, help="directory containing the preprocessed data")
    ap.add_argument("-c", nargs=1, help="directory containing the counted triples")
    args = ap.parse_args()
    d = args.d[0]
    c = args.c[0]
    for curr in [c,d]:
        if not os.path.exists(curr):
            raise Exception("directory %s does not exist"%curr)
    
    for filename in glob.glob(os.path.join(d,"*")):
        print "processing file %s.."%filename
        df = process(filename)
        dst_filename = os.path.join(c,os.path.basename(filename))
        print "writing to %s.."%dst_filename
        handle = gzip.open(dst_filename, "wb")
        for i,row in df.iterrows():
            t = triple.Triple(*row)
            handle.write(t.totabs()+'\n')
if __name__=="__main__":
    main()

