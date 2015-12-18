# a couple of modules that might come in handy
from itertools import islice, izip, chain, permutations
import io, sys
import argparse
import time
import datetime
from collections import Counter, defaultdict
from multiprocessing import Pool
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import gzip
import md5

def main():
	parser = argparse.ArgumentParser(description='Find alignments of bilingual corpus using EM training')
	parser.add_argument('-i','--input', type=str, help='Pre-processed training data', required=True)
	parser.add_argument('-m','--model', type=str, help='Model used for training', choices=['0-Rooth', '0-OConnor', '1-OConnor'], default='0-Rooth', required=False)
	parser.add_argument('-f','--frames', type=int, help='Number of frames to look for', default=10, required=False)
	parser.add_argument('-t','--threads', type=int, help='Number of threads', default=1, required=False)
	parser.add_argument('-o','--output', type=str, help='Output', default='', required=False)
	parser.add_argument('-it','--iterations', type=int, help='Number of iterations', default=10, required=False)
	parser.add_argument('-p', '--plot', type=bool, help='Plot likelihood per iteration (only available for model 0-Rooth', default=False, required=False)
	args = vars(parser.parse_args())


	global check
	check = True

	global iterations
	iterations = args['iterations']

	global frames
	frames = args['frames']

	global threads
	threads = args['threads']

	global tuples
	global counts
	global N
	tuples, counts = readData(args['input'])
        global counts_np
        print "converting the counts to numpy..."
        counts_np = np.array(counts)
	N = len(tuples)

	global voc
	global V
        print "extracting the vocabulary..."
	voc = extractVocabulary()
	V = len(voc)
	global vTuples
        print "encoding the tuples..."
	vTuples = encodeTuples()

        global vTuples_np
        print "converting the tuples in numpy..."
        vTuples_np = np.array(vTuples)

        global word_tuples_idx
        print "extracting the word indexes in the tuples..."
        word_tuples_idx = [
            [
                np.array(
                    # below: the 0 index is because a tuple of arrays,
                    # one for each dimension, is returned
                    # (the dimensionality is just 1)
                    np.where(vTuples_np[:,a] == w)[0]
                )
                for a
                in xrange(3)
            ]
            for w # the list has one element (list of tuple indices) for each word
            in xrange(V)
        ] # it's a list, not a matrix because the "rows" have different length (indices of occurrences)
        print "done."
	global beta
	beta = 0.001

	assigns = []
        print "starting inference.. the model you chose is %s"%args['model']
	if args['model'] == '0-Rooth':
		likelihoods, assigns = emTraining()
	if args['model']=='0-OConnor':
		assigns = lda()


	if args['output']:
		global output
		output = args['output']
		outputResults(assigns)
		if args['model']=='0-Rooth':
			outputLikelihoods(likelihoods)



	if bool(args['model']=='0-Rooth') & args['plot']:
		#likelihoods = np.array(likelihoods)
		#xnew = np.linspace(likelihoods.min(),likelihoods.max(),300)
		#ynew = spline(likelihoods,xnew)
		plt.plot(likelihoods)
		#'''
		for i in xrange(49):
			likelihoods, assigns = emTraining()
			plt.plot(likelihoods)
		#'''
		plt.ylabel('Log Likelihood')
		plt.xlabel('Iteration')
		plt.show()

def outputResults(assigns):
	print 'Writing output...'
	results = [[] for f in xrange(frames)]
	for i in xrange(N):
		results[int(assigns[i])] += [tuples[i][0]]

	if not os.path.exists(output):
		os.makedirs(output)
	for f in xrange(frames):
		with open(output+'/frame '+str(f),'w') as out:
			r = list(set(results[f]))
			out.write('\n'.join(r))


def outputLikelihoods(likelihoods):
	with open(output+'/likelihoods', 'w') as out:
		out.write(str(likelihoods))

def readData(dataFile):
	print 'Reading data...'
	tuples = []
	counts = []

        if os.path.splitext(dataFile)[-1] == '.gz':
            openfunc = gzip.open
        else:
            openfunc = open

	with openfunc(dataFile, 'rU') as f:
		for entry in f:
			entry = entry.strip().split('\t')
                        _t = np.array(entry[0:3])
			tuples.append(_t)
                        c_raw = entry[3]
                        c = float(c_raw)
			counts.append(c)
			#break
        print "done."
	return np.array(tuples), counts


def extractVocabulary():
    flattened = tuples.flatten()
    words_set = set(flattened)
    vocabulary = list(words_set)
    return vocabulary

def encodeTuples():
    return np.array([
        [
            voc.index(tuples[i][a])
            for a
            in xrange(3)
        ]
        for i
        in xrange(N)
    ])

#'''
def estimatePosterior(phi, theta):
	global check
	pTable = np.array([np.array([np.zeros(frames) for a in xrange(3)]) for w in xrange(V)])
	for w in xrange(V):
		for a in xrange(3):
			for f in xrange(frames):
				pTable[w][a][f] = theta[f]*phi[f][a][w]
			# normalize
			num = sum(pTable[w][a])
			if num > 0:
				pTable[w][a] = pTable[w][a]/num
	return pTable
#'''


def estimatePosteriorNumpy(phi, theta):
	pTable = np.multiply(theta, phi.transpose(2,1,0))
	num = np.sum(pTable, axis=2)[:,:,np.newaxis]
	num[num==0] = 1 #prevent divison by 0
	return np.divide(pTable, num)


def maximizePhi(pTable):
	phi = np.array([np.array([np.zeros(V) for w in xrange(3)]) for f in xrange(frames)])
	for f in xrange(frames):
		for a in xrange(3):
			for w in xrange(V):
				for i in xrange(N):
					if vTuples[i][a] == w:
						phi[f][a][w] += counts[i]*pTable[w][a][f]
			# normalize
			phi[f][a] = phi[f][a]/sum(phi[f][a])
	return phi


def maximizePhiNumpy(pTableForN):
	c = np.multiply(counts, pTableForN.transpose(2,1,0)) #(f, a, i)
	phi = np.zeros((3,V,frames)) #(a, w, f)
	for a in xrange(3):
			phi[a,:,...] = np.array([np.sum(c[:,a, np.nonzero(vTuples[:,a]==w)[0] ], axis=1)
								for w in xrange(V)])
	phi = phi.transpose(2,0,1)
	num = np.expand_dims(np.sum(phi, axis=2),2)
	return phi/num


def maximizeTheta(pTable):
	theta = np.zeros(frames)
	for f in xrange(frames):
		theta[f] = sum([counts[i]*sum([pTable[vTuples[i][a]][a][f] for a in xrange(3)]) for i in xrange(N)])
	# normalize
	return theta/sum(theta)


def maximizeThetaNumpy(pTableForN):
	theta = np.multiply(counts, np.sum(pTableForN.transpose(2,0,1), axis=2))
	theta = np.sum(theta, axis=1)
	return theta/sum(theta)

def emLikelihood(pTable, phi, theta):
	ll = 0
	for i in xrange(N):
		count = counts[i]
		for a in xrange(3):
			w = vTuples[i][a]
			for f in xrange(frames):
				ll += count*pTable[w][a][f]*math.log(theta[f]*phi[f][a][w])
	return ll


def emLikelihoodNumpy(pTableForN, phi, theta):
	# C(i)*p(f|w_i^a)
	cPosForN = np.multiply(counts, pTableForN.transpose((2,1,0))) #(f, a, i)
	#log(theta(f)*phi(w))
	thetaPhi = np.multiply(theta, phi.transpose(2,1,0)) #(f, a, w)
	thetaPhiForN = np.log(thetaPhi[list(vTuples), range(3)]) #(f, a, i)

	ll = np.multiply(cPosForN.transpose(2,1,0), thetaPhiForN)
	return np.sum(ll)


def chooseAssignmentsEM(phi):
	print 'Choosing assignments...'
	assigns = np.zeros(N)
	for i in xrange(N):
		argmaxF = [0, None]
		for f in xrange(frames):
			pf = np.prod([phi[f][a][vTuples[i][a]] for a in xrange(3)])
			#pf = np.prod([phi[f][a][data[i][2][a]] for a in xrange(3)])
			if pf > argmaxF[0]:
				argmaxF = [pf, f]
		assigns[i] = argmaxF[1]
	print 'Completed choosing assignments...'
	return assigns


def chooseAssignmentsEMNumpy(phi):
	print 'Choosing assignments...'
	phiPerN = phi.transpose(2,1,0)[list(vTuples), range(3)] #(i, a, f)
	phiProd = np.prod(phiPerN.transpose(0,2,1), axis=2) #(i, f)
	assigns = np.argmax(phiProd, axis=1)
	print 'Completed choosing assignments...'
	return assigns

def emTraining():
	print 'Starting EM training...'
	globalStart = time.time()

	print 'Initializing...'
	# random initialization
	'''
	phi = np.zeros((frames,3,V))
	for f in xrange(frames):
		for a in xrange(3):
			p = np.array([np.random.uniform(0,1) for w in xrange(V)])
			phi[f][a] = p/sum(p)
	'''
	phi = np.random.dirichlet(np.ones(V), (frames,3))

	# uniform initialization
	theta = np.array([float(1)/frames for f in xrange(frames)])

	prevLL = float('-infinity')
	likelihoods = []
	for it in xrange(1,(iterations+1)):
		print '\tIteration', str(it)
		start = time.time()



		pTable = estimatePosteriorNumpy(phi, theta)
		#print 'p(f|w) equal:',np.array_equal(pTable, estimatePosterior(phi, theta))


		pTableForN = pTable[list(vTuples), range(3)] #(i, a, f)
		#st = time.time()
		phi = maximizePhiNumpy(pTableForN)
		#print getDuration(st, time.time())
		#st = time.time()
		#phi = maximizePhiNumpy(pTable)
		#print getDuration(st, time.time())
		#print 'phi(w) equal:',np.array_equal(phi, maximizePhi(pTable))
		#2/0

		#st = time.time()
		theta = maximizeThetaNumpy(pTableForN)
		#print getDuration(st, time.time())
		#st = time.time()
		#theta = maximizeThetaNumpy(pTable)
		#print getDuration(st, time.time())

		#print 'theta(f) equal:',np.array_equal(theta, maximizeTheta(pTable))

		#2/0

		print '\t\tIteration completed in', getDuration(start, time.time())
		#ll = emLikelihood(pTable, phi, theta)
		#print '\t\tLog Likehood:', str(ll)
		#st = time.time()
		#ll_o = emLikelihood(pTable, phi, theta)
		#print getDuration(st, time.time())
		#st = time.time()
		ll = emLikelihoodNumpy(pTableForN, phi,theta)
		likelihoods.append(ll)
		#print getDuration(st, time.time())

		#print '\t\tLog Likehood (other):', str(ll_o)
		print '\t\tLog Likehood:', '%.10f' % ll


		print '>', ll > prevLL
		#print '=',ll == prevLL
		prevLL = ll
	print 'EM training completed in', getDuration(globalStart, time.time())


	#'''
	#st = time.time()
	#assigns_o = chooseAssignmentsEM(phi)
	#print getDuration(st, time.time())

	#st = time.time()
	assigns = chooseAssignmentsEMNumpy(phi)
	#print getDuration(st, time.time())
	#print 'assignments equal:',np.array_equal(assigns_o, assigns)

	return likelihoods, assigns
	#'''

def posteriorLDA(fwCounts, fCounts, i, a):
	fProbs = np.zeros(frames)
	wa = vTuples[i][a]
	for f in xrange(frames):
 		fProbs[f] = (((beta+fwCounts[f][wa])/(V*beta+fCounts[f]))
						* (fCounts[f]/sum(fCounts)))
	num = sum(fProbs)
	return [fProbs[f]/num for f in xrange(frames)]


def chooseAssignmentsLDA(fwCounts, fCounts):
	assigns = np.zeros(N)
	for i in xrange(N):
		fProbs = [posteriorLDA(fwCounts, fCounts, i, a) for a in xrange(3)]
		fProbs = zip(*fProbs) # group by frame
		fProbs = [np.prod(fProbs[f]) for f in xrange(frames)]
		assigns[i] = fProbs.index(max(fProbs))
	return assigns


def chooseAssignmentsLDANumpy(fwCounts, fCounts):
	posterior = 0


def fwCountsThreadNumpy((assigns, globalFs)):

    localF = len(globalFs)
    print "calculating frame_tuples_idx..."
    frame_tuples_idx = [
        [
            set([
                i
                for i
                in xrange(len(counts_np))
                if assigns[i][a] == f
            ])
            for f
            in xrange(localF)
        ]
        for a
        in xrange(3)
    ]
    print "done. now calculating the fwCounts..."
    fwCounts = [
        [
            np.sum(counts_np[
                # this list is a list of tuple indexes
                # basically I need a union of intersections
                list(reduce(lambda s1,s2: s1.union(s2),
                    [
                        # those indexes are the indexes associated to the word
                        set(word_tuples_idx[w][a])
                            # but they also need to be associated to the frame
                            .intersection(frame_tuples_idx[a][f])
                        for a
                        in xrange(3)
                    ]
                ))
            ])
            for w
            in xrange(V) # 1 row for every word in the vocabulary
        ]
        for f
        in xrange(localF) # one column for every frame
    ]
    print "done."
    return fwCounts

def fwCountsThread((assigns, globalFs)):
        print "fwCountsThread started.."
	localF = len(globalFs)
	fwCounts = [
            [
                sum(
                    sum(
                        float(counts[i])
                        for a
                        in xrange(3)
                        if vTuples[i][a]==w
                            and assigns[i][a]==globalFs[f]
                    )
                    for i
                    in xrange(N)
                )
                for w
                in xrange(V)
            ]
            for f
            in xrange(localF)
        ]
        print "fwCountsThread returning."
	return fwCounts

def _hash(stuff):
    """
    returns a checksum (hex string format) of stuff
    """
    if type(stuff) is np.ndarray:
        s = stuff.tostring()
    elif type(stuff) is list:
        s = str(stuff)
    return md5.md5(s).hexdigest()

def lda(prior=False):
	print 'Starting LDA...'
	globalStart = time.time()

	# 1. initialize randomly
	print 'Initializing...'
	assigns = [
            [
                np.random.randint(frames)
                for a
                in xrange(3)
            ]
            for i
            in xrange(N)
        ]
	#C(f,w)
        _d = [] # debug variable FIXME delete it
        # here I am testing both functions
        # FIXME of course this needs to be changed
        for f in [fwCountsThread, fwCountsThreadNumpy]:
            start = time.time()
            p = Pool(threads)
            n = int(math.ceil(frames/float(threads)))
            args = zip(
                [assigns]*threads,
                [
                    range(frames)[i:i+n]
                    for i
                    in xrange(0, frames, n)
                ]
            )
            fwCountsMap = p.map(f, args)
            p.close()
            fwCounts = list(chain.from_iterable(fwCountsMap))
            print '\t* C(f,w) calculated in', getDuration(start, time.time()), '*'
            _d.append(fwCounts)
	# C(f)
	fCounts = [
            sum(
                sum(
                    float(data[i][1])
                    for a
                    in xrange(3)
                    if assigns[i][a] == f
                )
                for i
                in xrange(N)
            )
            for f
            in xrange(frames)
        ]

	t1 = chooseAssignmentsLDA(fwCounts, fCounts)
	t2 = chooseAssignmentsLDANumpy(fwCounts, fCounts)

	perms = list(permutations(range(3)))
	P = len(perms)
	# 3. repeat long enough
	for it in xrange(1, (iterations+1)):
		print '\tIteration', str(it)
		start = time.time()

		for i in xrange(N):
			# 2. randomly choose a variable and draw it's new value from
			# the respective conditional probability
			for a in perms[np.random.randint(P)]:
				# update counts to reflect removal of w_i^a's assignment
				fwCounts[assigns[i][a]][vTuples[i][a]] -= counts[i]
				fCounts[assigns[i][a]] -= counts[i]

				# assign new frame to w_i^a
				fProbs = posteriorLDA(fwCounts, fCounts, i, a)
				assigns[i][a] = fProbs.index(max(fProbs))

				# update counts to reflect assignment of the new frame
				fwCounts[assigns[i][a]][vTuples[i][a]] += counts[i]
				fCounts[assigns[i][a]] += counts[i]

		print '\t\tIteration completed in', getDuration(start, time.time())
	print 'LDA completed in', getDuration(globalStart, time.time())

	# 4. choose the last assignment as a sample
	return chooseAssignmentsLDA(fwCounts, fCounts)


def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))


if __name__ == '__main__':
    main()
