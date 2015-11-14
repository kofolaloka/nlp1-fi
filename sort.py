import pandas as pd
import argparse
import os
import glob

# local imports
import triple
import utils

def main():
    in_d, out_d,_ = utils.argsdirs("Sorting")
    
    for filename in utils.filenames(in_d):
        tome_in = triple.Tome(filename)
        filename_out = utils.new_filename(out_d,filename)
        tome_out = triple.Tome(filename_out)
        writer = tome_out.writer()
        for tr in tome_in.sort():
            writer(tr)
    
if __name__=="__main__":
    main()
