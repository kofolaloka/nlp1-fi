from os import listdir, path, remove
import argparse

def main():
    commandline_parser = argparse.ArgumentParser("Evaluation of semantic frame induction")

    commandline_parser.add_argument("--clusters", nargs =1, help="Specifies the path to the directory where the induced frames are stored")

    commandline_parser.add_argument("--framenet", nargs =1, help="Specifies the path to the directory where the framenet clusters are stored")
    commandline_parser.add_argument("--output-folder", nargs=1, help="Specifies the path to the directory where the ouput evaluation files are stored.")

    args = vars(commandline_parser.parse_args())
    clusters_dir = args["corpus_directory"][0]
    fn_dir = args["test_set_directory"][0]
    output_folder = args["threshold"][0]
    filter_clusters(clusters_dir)
    max_match(clusters_dir, fn_dir, output_folder)
    vocab_clusters = vocab(clusters_dir)
    vocab_fn = vocab(fn_dir)
    print 'Coverage: '+ coverage(vocab_clusters, vocab_fn)

def vocab(folder):
    vocab = set()
    for file in listdir(folder):
        file_path = path.join(folder, file)
        file = open(file_path, 'r')
        for line in file:
            vocab.add(line)
    return vocab

def coverage(vocab_clusters, vocab_fn):
    ratio = len(vocab_clusters, vocab_fn)
    return ratio

def dice_similarity(set1, set2):
    sim = (2 * len(set1.intersection(set2)))/(len(set1) + len(set2))
    return sim

def filter_clusters(clusters_dir):
    for frame in listdir(clusters_dir):
        frame_id = frame
        frame_path = path.join(clusters_dir, frame)
        frame = frame.open(frame_path, 'r')
        i = 0
        for line in frame:
            i += 1
        if i < 5:
            remove(frame_path)
        else:
            frame_path.close()

def max_match(clusters_dir, fn_dir, output_folder):
    '''
    @param clusters_dir: clusters induced
    @param fn_dir: framenet clusters
    @param output_folder: output folder
    @return: file clusters_id \t frame_id
    dice sim
    '''
    frames = listdir(clusters_dir)
    fn_frames = listdir(fn_dir)
    output_path = path.join(output_folder, 'max_match_clusters')
    output_file = open(output_path, 'w')
    for frame in frames:
        frame_id = frame
        frame_path = path.join(clusters_dir, frame)
        frame = open(frame_path, 'r')
        wordset = set()
        for line in frame:
            wordset.add(line)
        sims = {}
        for fn_frame in fn_frames:
            frame_fn_id = fn_frame
            frame_fn_path = path.join(fn_dir, fn_frame)
            frame_fn = open(frame_fn_path, 'r')
            wordset_fn = set()
            for line in fn_frame:
                wordset_fn.add(line)
            sim = dice_similarity(wordset, wordset_fn)
            sims[frame_fn_id] = sim
        max_match_frame = max(sims).key
        output_file.write(frame_id + '\t' + max_match_frame + '\t' + sim[max_match_frame])

if __name__ == '__main__':
    main()