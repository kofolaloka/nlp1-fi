import pandas as pd
import os
import glob

# local imports
import triple
import utils

def main():
    in_d, out_d, members_groupby = utils.argsdirs("Counting the triples")
    
    for filename in utils.filenames(in_d):
        print "processing file %s.."%filename
        tome_in = triple.Tome(filename)
        filename_out = utils.new_filename(out_d,filename)
        print "writing to %s.."%filename_out
        tome_out = triple.Tome(filename_out)
        
        writer = tome_out.writer()
        for tr in tome_in.group_sum(members_groupby):
            writer(tr)

if __name__=="__main__":
    main()

