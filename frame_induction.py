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

def main():
	parser = argparse.ArgumentParser(description='Find alignments of bilingual corpus using EM training')
	parser.add_argument('-i','--input', type=str, help='Pre-processed training data', required=True)
	parser.add_argument('-m','--model', type=str, help='Model used for training', choices=['0-Rooth', '0-OConnor', '1-OConnor'], default='0-Rooth', required=False)
	parser.add_argument('-f','--frames', type=int, help='Number of frames to look for', default=10, required=False)
	parser.add_argument('-t','--threads', type=int, help='Number of threads', default=1, required=False)

	args = vars(parser.parse_args())

	global frames
	frames = args['frames']

	global threads
	threads = args['threads']

	global data
	global N
	data = readData(args['input'])
	N = len(data)

	global voc
	global V
	voc = extractVocabulary(data)
	V = len(voc)
	data = extendData()

	global beta
	beta = 0.1

	assigns = []
	if args['model'] == '0-Rooth':
		assigns = emTraining()
	if args['model'] == '0-OConnor':
		assigns = lda()


def readData(dataFile):
	print 'Reading data...'
	data = []
	with open(dataFile, 'rU') as f:
		for entry in f:
			entry = entry.strip().split('\t')
			data.append([entry[0:3], float(entry[3])])
			#break
	return data


def extractVocabulary(data):
	vocabulary = []
	for triple, count in data:
		vocabulary += triple
	return list(set(vocabulary))


def extendData():
	return [data[i]+[[voc.index(data[i][0][a]) for a in xrange(3)]] for i in xrange(N)]


def estimatePosterior(phi, theta):
	pTable = [np.zeros(frames) for w in xrange(V)]
	for i in xrange(N):
		for a in xrange(3):
			w = data[i][2][a]
			for f in xrange(frames):
				pTable[w][f] += theta[f]*phi[f][a][w]
	# normalize
	for w in xrange(V):
		num = sum(pTable[w])
		pTable[w] = [pTable[w][f]/num for f in xrange(frames)]
	return pTable


def maximizePhi(pTable):
	p = Pool(threads)
	n = int(math.ceil(frames/float(threads)))
	args = zip([pTable]*threads, [range(frames)[i:i+n] for i in xrange(0, frames, n)])
	phi = p.map(maximizePhiThread, args)
	return list(chain.from_iterable(phi))


def maximizePhiThread((pTable, globalFs)):
	localF = len(globalFs)
	phi = [[np.zeros(V) for a in xrange(3)] for f in xrange(localF)]
	for f in xrange(localF):
		for a in xrange(3):
			for w in xrange(V):
				for i in xrange(N):
					if data[i][2][a] == w:
						phi[f][a][w] += data[i][1]*pTable[w][globalFs[f]]
			# normalize
			num = sum(phi[f][a])
			phi[f][a] = [phi[f][a][w]/num for w in xrange(V)]
	return phi


def maximizeTheta(pTable):
	p = Pool(threads)
	n = int(math.ceil(frames/float(threads)))
	args = zip([pTable]*threads, [range(frames)[i:i+n] for i in xrange(0, frames, n)])
	thetaMap = p.map(maximizeThetaThread, args)
	thetaList = list(chain.from_iterable(thetaMap))
	# normalize
	num = sum(thetaList)
	theta = [thetaList[f]/num for f in xrange(frames)]
	return theta

def maximizeThetaThread((pTable, globalFs)):
	localF = len(globalFs)
	theta = np.zeros(localF)
	for f in xrange(localF):
		theta[f] = sum(data[i][1]*sum(pTable[w][globalFs[f]] for w in data[i][2]) for i in xrange(N))
	return theta


def emLikelihood(pTable, phi, theta):
	ll = 0
	for i in xrange(N):
		count = data[i][1]
		for a in xrange(3):
			w = data[i][2][a]
			for f in xrange(frames):
				ll += count*pTable[w][f]*math.log(theta[f]*phi[f][a][w])
	return ll

def chooseAssignmentsEM(phi):
	print 'Choosing assignments...'
	assigns = np.zeros(N)
	for i in xrange(N):
		argmaxF = [0, None]
		for f in xrange(frames):
			pf = np.prod(phi[f][a][data[i][2][a]] for a in xrange(3))
			if pf > argmaxF[0]:
				argmaxF = [pf, f]
		assigns[i] = argmaxF[1]
	print 'Completed choosing assignments...'
	return assigns


def emTraining(iterations=2):
	print 'Starting EM training...'
	globalStart = time.time()

	print 'Initializing...'
	# random initialization
	phi = [[np.random.dirichlet(np.ones(V),size=1)[0] for wa in xrange(3)] for f in xrange(frames)]
	# uniform initialization
	theta = [float(1)/frames for f in xrange(frames)]

	for it in xrange(1,(iterations+1)):
		print '\tIteration', str(it)
		start = time.time()

		pTable = estimatePosterior(phi, theta)
		phi = maximizePhi(pTable)
		theta = maximizeTheta(pTable)

		print '\t\tIteration completed in', getDuration(start, time.time())
		print '\t\tLog Likehood:', str(emLikelihood(pTable, phi, theta))
	print 'EM training completed in', getDuration(globalStart, time.time())

	return chooseAssignmentsEM(phi)


def posteriorLDA(fwCounts, fCounts, i, a):
	fProbs = np.zeros(frames)
	wa = data[i][2][a]
	for f in xrange(frames):
		fProbs[f] = (((beta+fwCounts[f][wa])/(beta+fCounts[f]))
						* (fCounts[f]/sum(fCounts)))
	num = sum(fProbs)
	return [fProbs[f]/num for f in xrange(frames)]


def chooseAsignmentsLDA(fwCounts, fCounts):
	assigns = np.zeros(N)
	for i in xrange(N):
		fProbs = [posteriorLDA(fwCounts, fCounts, i, a) for a in xrange(3)]
		fProbs = zip(*fProbs) # group by frame
		fProbs = [np.prod(fProbs[f]) for f in xrange(frames)]
		assigns[i] = fProbs.index(max(fProbs))
	return assigns


def fwCountsThread((assigns, globalFs)):
	localF = len(globalFs)
	fwCounts = [[sum(sum(float(data[i][1]) for a in xrange(3)
				if (data[i][2][a] == w and assigns[i][a] == globalFs[f])) for i in xrange(N))
					for w in xrange(V)]
						for f in xrange(localF)]
	return fwCounts


def lda(prior=False, iterations=4):
	print 'Starting LDA...'
	globalStart = time.time()

	# 1. initialize randomly
	print 'Initializing...'
	assigns = [[np.random.randint(frames) for a in xrange(3)] for i in xrange(N)]
	#C(f,w)
	start = time.time()
	p = Pool(threads)
	n = int(math.ceil(frames/float(threads)))
	args = zip([assigns]*threads, [range(frames)[i:i+n] for i in xrange(0, frames, n)])
	fwCountsMap = p.map(fwCountsThread, args)
	fwCounts = list(chain.from_iterable(fwCountsMap))
	print '\t* C(f,w) calculated in', getDuration(start, time.time()), '*'
	# C(f)
	fCounts = [sum(sum(float(data[i][1]) for a in xrange(3) if assigns[i][a] == f) for i in xrange(N))
					for f in xrange(frames)]

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
				fwCounts[assigns[i][a]][data[i][2][a]] -= data[i][1]
				fCounts[assigns[i][a]] -= data[i][1]

				# assign new frame to w_i^a
				fProbs = posteriorLDA(fwCounts, fCounts, i, a)
				assigns[i][a] = fProbs.index(max(fProbs))

				# update counts to reflect assignment of the new frame
				fwCounts[assigns[i][a]][data[i][2][a]] += data[i][1]
				fCounts[assigns[i][a]] += data[i][1]

		print '\t\tIteration completed in', getDuration(start, time.time())
	print 'LDA completed in', getDuration(globalStart, time.time())

	# 4. choose the last assignment as a sample
	return chooseAsignmentsLDA(fwCounts, fCounts)


def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))


if __name__ == '__main__':
    main()
