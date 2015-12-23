from os import listdir, path, remove, mkdir, stat
import argparse

def main():
    commandline_parser = argparse.ArgumentParser("Evaluation of semantic frame induction")

    commandline_parser.add_argument("--clusters", nargs =1, help="Specifies the path to the directory where the induced frames are stored")

    commandline_parser.add_argument("--framenet", nargs =1, help="Specifies the path to the directory where the framenet clusters are stored")
    commandline_parser.add_argument("--output-folder", nargs=1, help="Specifies the path to the directory where the ouput evaluation files are stored.")

    args = vars(commandline_parser.parse_args())
    clusters_dir = args["clusters"][0]
    fn_dir = args["framenet"][0]
    output_folder = args["output_folder"][0]
    filter_clusters(clusters_dir)
    results = path.join(output_folder, 'results.txt')
    results = open(results, 'w')
    max_match(clusters_dir, fn_dir, output_folder, results)
    vocab_clusters = vocab(clusters_dir)
    vocab_fn = vocab(fn_dir)
    results.write('Coverage: '+ str(coverage(vocab_clusters, vocab_fn))+ '\n')
    results.close()

def average_sim(output_path, results):
	output = open(output_path, 'r')
	sum_sims = 0
	lines = 0
	for line in output:
		lines = lines + 1
		sim = float(line.split('\t')[4])
		sum_sims = sum_sims + sim
	average = sum_sims*1.0/lines*1.0
	results.write('Average sim: '+ str(average) + '\n')

def vocab(folder):
    vocab = set()
    for file in listdir(folder):
        file_path = path.join(folder, file)
        file = open(file_path, 'r')
        for line in file:
            verb = line.strip('\n')
            vocab.add(verb)
    return vocab

def coverage(vocab_clusters, vocab_fn):
    ratio = len(vocab_clusters)*1.0/ len(vocab_fn)*1.0
    return ratio

def dice_similarity(set1, set2):
    intersection = set1.intersection(set2)
    total = len(set1) + len(set2)
    sim = float(2 * len(intersection))/float(total)
    return sim

def filter_clusters(clusters_dir):
    for frame in listdir(clusters_dir):
        frame_id = frame
        frame_path = path.join(clusters_dir, frame)
        frame = open(frame_path, 'r')
        i = 0
        for line in frame:
            i += 1
        if i < 5:
            remove(frame_path)
        else:
            frame.close()

def max_match(clusters_dir, fn_dir, output_folder, results):
    '''
    @param clusters_dir: clusters induced
    @param fn_dir: framenet clusters
    @param output_folder: output folder
    @return: file clusters_id \t frame_id
    dice sim
    '''
    frames = listdir(clusters_dir)
    fn_frames = listdir(fn_dir)
    sum_verbs = 0
    output_path = path.join(output_folder, 'max_match_clusters')
    output_file = open(output_path, 'w')
    fn_wordsets = {}
    for frame in frames:
        frame_id = frame
        frame_path = path.join(clusters_dir, frame)
        frame = open(frame_path, 'r')
        wordset = set()
        for line in frame:
	    verb = line.strip('\n')
            wordset.add(verb)
        sims = {}
        for fn_frame in fn_frames:
            frame_fn_id = fn_frame.strip('.txt')
            frame_fn_path = path.join(fn_dir, fn_frame)
            frame_fn = open(frame_fn_path, 'r')
            wordset_fn = set()
            for line in frame_fn:
		verb = line.strip('\n')
                wordset_fn.add(verb)
	    fn_wordsets[frame_fn_id] = wordset_fn
            sim = dice_similarity(wordset, wordset_fn)
            sims[frame_fn_id] = sim
	    frame_fn.close()
	inverse = [(value, key) for key, value in sims.items()]
	max_match_frame = max(inverse)[1]
	max_match_frame_wordset = fn_wordsets[max_match_frame]
        max_match_frame_value = max(inverse)[0]
	overlap = wordset.intersection(max_match_frame_wordset)
	common = len(overlap)
        output_file.write(frame_id + '\t'+ str(len(wordset))+ '\t' +  max_match_frame + '\t'+ str(len(max_match_frame_wordset))+ '\t'+ str(max_match_frame_value)+ '\t'+ str(common)+ '\n')
	frame.close()
	overlap = []
	overlap_file = frame_id + '_' + max_match_frame
	overlap_file = path.join(output_folder, overlap_file)
	overlap_file = open(overlap_file, 'w')
	overlap = wordset.intersection(max_match_frame_wordset)
	common = len(overlap)
	for verb in overlap:
		overlap_file.write(verb + '\n')
        overlap_file.close()
	sum_verbs = sum_verbs + len(wordset)
    average_verb = sum_verbs*1.0/len(frames)*1.0
    results.write('Average verbs for frame: ' + str(average_verb)+ '\n')
    output_file.close()    
    average_sim(output_path, results)
    order(output_folder, output_path, overlap_file)

def order(output_folder, output_path, output_file):
	overlap_order = open(path.join(output_folder, 'max_match_overlap.txt'), 'w')
	sim_order = open(path.join(output_folder, 'max_match_sim.txt'), 'w')
	data = []
	output_file = open(output_path, 'r')
	for line in output_file:
		entries = line.split('\t')
		entries[5] = entries[5].replace('\n', '')
		entries[5] = int(entries[5])
		entries[4] = float(entries[4])
		data.append(entries)
	data_overlap = sorted(data, key = lambda x:x[5], reverse=True)
	data_sim = sorted(data, key = lambda x:x[4], reverse=True)
	for entry in data_overlap:
		for i in entry:
			overlap_order.write(str(i) + '\t')
		overlap_order.write('\n')
	for entry in data_sim:
		for i in entry:
			sim_order.write(str(i) + '\t')
		sim_order.write('\n')
	overlap_order.close()
	sim_order.close()
	output_file.close()

if __name__ == '__main__':
    main()