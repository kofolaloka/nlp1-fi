import xml.etree.ElementTree as ET
import gzip
from os import listdir, remove, path, remove
import argparse

def extract_clusters(framenet, output):
	for file in listdir(framenet):
		folder_path = path.join(framenet, file)
		if file == 'frame':
			for f in listdir(folder_path):
				file_path = path.join(folder_path, f)
				name = f.strip(".xml")
				name = name + ".txt"
				output_path = path.join(output, name)
				output_file = open(output_path, 'w')
				tree_xml = ET.parse(file_path)
				root = tree_xml.getroot()
				for child in root:
					if child.tag == "{http://framenet.icsi.berkeley.edu}lexUnit":
						if child.attrib['POS'] == 'V':
							verb = child.attrib['name'].strip('.v')
							output_file.write(verb + '\n')
				output_file.close()

def clean(output_dir):
	for f in listdir(output_dir):
		file_path = path.join(output_dir, f)
		f = open(file_path, 'r')
		content = ""
		for line in f:
			content = content + line
		if content == '':
			remove(file_path)
		else:
			f.close()

def main():

    commandline_parser = argparse.ArgumentParser("Add description.")
     
    commandline_parser.add_argument("--framenet-directory", nargs =1, help="Specifies the path to the directory where the corpus is stored.")
     
    commandline_parser.add_argument("--output-directory", nargs=1, help="Specifies the path to the directory where the test files are stored.")
     
    args = vars(commandline_parser.parse_args())
    fn_dir = args["framenet_directory"][0]
    output_dir = args["output_directory"][0]
    extract_clusters(fn_dir, output_dir)
    clean(output_dir)

if __name__ == '__main__':
    main()    