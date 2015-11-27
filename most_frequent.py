import pandas as pd
import argparse
import os
import glob

# local imports
import triple
import utils

def main():
    in_d, out_d,m,n = utils.argsdirs("Most frequent triples",["n"])
    n = int(n)
    tomes = [
        triple.Tome(filename)
        for filename
        in utils.filenames(in_d)
    ]
    
    filename_out = utils.new_filename(out_d,"most_frequent.gz")
    tome_out = triple.Tome(filename_out)
    
    print "joining the tomes.."
    tome_join = triple.Tome(tomes)
    print "grouping/summing (again).."
    tome_join = tome_join.group_sum(m)
    print "sorting the tomes (again).."
    tome_join = tome_join.sort()
    print "getting the first %d.."%n
    tome_join = tome_join.first(n)
    print "writing everything down.."
    writer = tome_out.writer()
    for tr in tome_join:
        writer(tr)
    print "done."

if __name__=="__main__":
    main()

