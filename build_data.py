import gzip
import argparse
from os import listdir, path, makedirs
from nltk.stem.wordnet import WordNetLemmatizer
import string

# local imports
import triple
import utils

object_tags = [ 'dobj' ]
subject_tags = [ 'nsubj', 'nsubjpass', 'csubj', 'csubjpass' ]
lemmatizer = WordNetLemmatizer()

def infinitive(verb):
    try:
        #verb = unicode(verb,"utf-8")
        filtered = filter(lambda x: x in string.printable, verb)
        ret = lemmatizer.lemmatize(filtered, 'v')
    except Exception as e:
        print "Exception: %s ; unable to lemmatize verb form %s"%(str(e),verb)
        import ipdb; ipdb.set_trace()
    return ret

def preprocess(file_path, output_path):
    input_file = gzip.open(file_path, 'rb')
    output_tome = triple.Tome(output_path)
    writer = output_tome.writer()

    for line in input_file:
        items = line.strip().split()
        v = items[0]
        s = ''
        o = ''
        for item in items[1:]:
            e = item.split('/')
            if len(e) == 4:
                if e[2] in object_tags:
                    o = e[0]
                elif e[2] in subject_tags: 
                    s = e[0]
            else:
                value = e[0]
                break
        if '' not in [v,s,o]:
            v = infinitive(v)
            t = triple.Triple(v,s,o,value)
            writer(t)
    input_file.close()
    print file_path,'done'   
    
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
    files = utils.filenames(data_folder)
    for file_path in files:
        output_path = utils.new_filename(output_folder, file_path)
        preprocess(file_path, output_path)
    
if __name__ == '__main__':
    main()
