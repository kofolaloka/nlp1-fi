# a couple of modules that might come in handy
from itertools import islice, izip, chain
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
	args = vars(parser.parse_args())

	global frames
	frames = args['frames']

	global data
	global N
	data = readData(args['input'])
	N = len(data)

	global voc
	global V
	voc = extractVocabulary(data)
	V = len(voc)
	data = extendData()

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
				pTable[w][f] = theta[f]*phi[f][a][w]
	# normalize
	for w in xrange(V):	
		num = sum(pTable[w])
		pTable[w] = [pTable[w][f]/num for f in xrange(frames)]
	return pTable


def maximizePhi(pTable):
	phi = [[np.zeros(V) for a in xrange(3)] for f in xrange(frames)]
	for f in xrange(frames):
		for a in xrange(3):
			for w in xrange(V):
				for i in xrange(N):
					if data[i][2][a] == w:	
						phi[f][a][w] += data[i][1]*pTable[w][f]
			# normalize
			num = sum(phi[f][a])
			phi[f][a] = [phi[f][a][w]/num for w in xrange(V)]
	return phi


def maximizeTheta(pTable):
	theta = np.zeros(frames)
	for f in xrange(frames):
		theta[f] = sum(data[i][1]*sum(pTable[w][f] for w in data[i][2]) for i in xrange(N))
	# normalize
	num = sum(theta)
	theta = [theta[f]/num for f in xrange(frames)]
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


def emTraining(iterations=10):
	print 'Starting EM training...'
	globalStart = time.time()

	print 'Initializing...'
	# random initialization
	phi = [[np.random.dirichlet(np.ones(V),size=1)[0] for wa in xrange(3)] for f in xrange(frames)]
	# uniform initialization
	theta = [float(1)/frames for f in xrange(frames)]

	for i in xrange(1,(iterations+1)):
		print '\tIteration', str(i)
		start = time.time()

		pTable = estimatePosterior(phi, theta) 
		phi = maximizePhi(pTable)
		theta = maximizeTheta(pTable)

		print '\t\tIteration completed in', getDuration(start, time.time())
		print '\t\tLog Likehood:', str(emLikelihood(pTable, phi, theta))
	print 'EM training completed in', getDuration(globalStart, time.time())

	return chooseAssignmentsEM(phi)


def lda(beta=0.1, prior=False, interations=10):
	print 'Starting LDA...'
	globalStart = time.time()

	# 1. initialize randomly
	assigns = [[np.random.randint(frames) for a in xrange(3)] for i in xrange(N)]
	
	# 3. repeat long enough
	for i in xrange(1, (iterations+1)):
		print '\tIteration', str(i)
		# 2. randomly choose a variable and draw it's new value from 
		# the respective conditional probability 

		print '\t\tIteration completed in', getDuration(start, time.time())
	
	
	print 'LDA completed in', getDuration(globalStart, time.time())

	# 4. choose the last assignment as a sample
	return assigns

def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))


if __name__ == '__main__':
    main()