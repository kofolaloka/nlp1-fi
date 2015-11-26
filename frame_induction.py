# a couple of modules that might come in handy
from itertools import islice, izip
import io, sys
import argparse
import time
import datetime
from collections import Counter, defaultdict
from multiprocessing import Pool
import math

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

	if args['model'] == '0-Rooth':
		results = emTraining(data)

def readData(dataFile):
	print 'Reading data...'
	data = []
	with open(dataFile, 'rU') as f:
		for entry in f:
			entry = entry.strip().split('\t')
			data.append([entry[0:3], float(entry[3])])
			#break
	return data

def estimatePosterior(phi, theta):
	pTable = [[[float(1)/frames]*frames]*3 for i in xrange(N)]
	for i in xrange(N):
		for w in xrange(3):
			for f in xrange(frames):
				pTable[i][w][f] = theta[i][f]*phi[f][w]
	return pTable

def maximizePhi(pTable):
	phi = [range(3) for f in xrange(frames)]
	for f in xrange(frames):
		for w in xrange(3):
			phi[f][w] = sum(data[i][1]*pTable[i][w][f] for i in xrange(N))
		num = sum(phi[f])
		phi[f] = [phi[f][w]/num for w in xrange(3)]
	return phi

def maximizeTheta(pTable):
	theta = [range(frames) for i in xrange(N)]
	for i in xrange(N):
		for f in xrange(frames):
			theta[i][f] = sum(data[i][1]*sum(pTable[i][w][f] for w in xrange(3)) for i in xrange(N))
		num = sum(theta[i])
		theta[i] = [theta[i][f]/num for f in xrange(frames)]
	return theta

def emLikelihood(pTable, phi, theta):
	ll = 0
	for i in xrange(N):
		for w in xrange(3):
			for f in xrange(frames):
				ll += data[i][1]*pTable[i][w][f]*math.log(theta[i][f]*phi[f][w])
	return ll

def emTraining(iters=10):
	print 'Beginning EM training...'
	globalStart = time.time()

	print 'Initializing...'
	# NxWxF
	pTable = [[[float(1)/frames for f in xrange(frames)] for w in xrange(3)] for i in xrange(N)]

	#print pTable[0]
	for i in xrange(1,11):
		print '\tIteration', str(i)
		start = time.time()

		phi = maximizePhi(pTable)
		theta = maximizeTheta(pTable)
		pTable = estimatePosterior(phi, theta) 

		print '\t\tIteration completed in', getDuration(start, time.time())
		print '\t\tLog Likehood:', str(emLikelihood(pTable, phi, theta))
	print 'EM training completed in', getDuration(globalStart, time.time())
	
	#print pTable[0]

def lda(prior=False):
	print 'Beginning LDA...'
	globalStart = time.time()
	print 'LDA completed in', getDuration(globalStart, time.time())

def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))

if __name__ == '__main__':
    main()