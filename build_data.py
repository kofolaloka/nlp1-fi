import gzip
import argparse
from os import listdir, path, makedirs

def preprocess(file_path, output_path):
    input_file = gzip.open(file_path, 'rb')
    output_file = gzip.open(output_path, 'wb')
    for line in input_file:
        items = line.strip().split()
        v = items[0]
        s = ''
        o = ''
        args = True
        i = 1
        while args == True:
            e = items[i].split('/')
            if len(e) == 4:
                if e[2] == 'dobj':
                    o = e[0]
                elif e[2] == 'nsubj' or e[2] == 'nsubjpass' or e[2] == 'csubj' or e[2] == 'csubjpass':
                    s = e[0]
            else:
                value = e[0]
                args = False
            i = i + 1
        if s != '' and v != '' and o != '':
            output_file.write(v + '\t'+ s + '\t'+ o + '\t'+ value + '\n')
    input_file.close()
    output_file.close() 
    print file_path + ' done'   
    
def main ():
    commandline_parser = argparse.ArgumentParser("Pre-processing of data")

    commandline_parser.add_argument("--data-folder", nargs =1, help="Specifies the path of the folder containing the data.")
    
    commandline_parser.add_argument("--output-folder", nargs =1, help="Specifies the path of the output folder.")
    
    args = vars(commandline_parser.parse_args())
    data_folder = args["data_folder"][0]
    output_folder = args["output_folder"][0]
    output_folder = path.join(output_folder,'dataset')
    if not path.exists(output_folder):
        makedirs(output_folder)
    files = listdir(data_folder)
    for f in files:
        file_path = path.join(data_folder, f)
        output_path = path.join(output_folder, f)
        preprocess(file_path, output_path)
    
if __name__ == '__main__':
    main()