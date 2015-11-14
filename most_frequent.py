import pandas as pd
import argparse
import os
import glob

# local imports
import triple
import utils

def main():
    in_d, out_d,_ = utils.argsdirs("Most frequent triples")
    
    tomes = [
        triple.Tome(filename).first(1000)
        for filename
        in utils.filenames(in_d)
    ]
    
    filename_out = utils.new_filename(out_d,"most_frequent.gz")
    tome_out = triple.Tome(filename_out)
    tome_join = triple.Tome(tomes).sort().first(1000)
    writer = tome_out.writer()
    for tr in tome_join:
        writer(tr)
    
if __name__=="__main__":
    main()

