import os
from unicodeMagic import UnicodeReader, UnicodeWriter
from lsaAlgorithm import LSAAlgo
from dictUtils import MyDict
from filters import *
import os
import csv
	
class ICSMTests():
	"""
		Grabs the output file of the LSA algorithm and compares it to the oracle.
		Appends the type of result to each row {tp, fp}.
		Also appends the missing results (fn) to the output file.
		Returns with the numbers for all types of results, including precision, recall and f-measure.
	"""
	def computeResults(self, dataPath, filename, outfilename, oracle, nameEmailData):

		from gensim import models, similarities, matutils, corpora, utils
		
		if os.path.exists(os.path.join(dataPath, filename)):

			indices = range(len(nameEmailData.keys()))
			
			documents = MyDict(os.path.join(dataPath, 'documents.dict'))
			directLookup = MyDict(os.path.join(dataPath, 'directLookup.dict')) #input data index -> document index
			index = similarities.docsim.Similarity.load(os.path.join(dataPath, 'index'))

			# Initial values
			tp = 0.0
			fp = 0.0
			fn = 0.0
			tn = 0.0
			
			matchedTuples = set()
			
			# Write all matched indexes to a set
			f = open(os.path.join(dataPath, filename), 'rb')
			reader = UnicodeReader(f)
			
			g = open(os.path.join(dataPath, outfilename), 'wb')
			writer = csv.writer(g, delimiter=';')
			for row in reader:
				header = row #Ignore header
				header.append('kind')
				writer.writerow(header)
				break
			for row in reader:	
				
				# Put the smallest value first.
				if int(row[0]) <= int(row[1]):
					idx1 = int(row[0])
					idx2 = int(row[1])
				else:
					idx2 = int(row[0])
					idx1 = int(row[1])
					
				matchedTuples.add((idx1, idx2))
				if oracle[(idx1, idx2)] == 1:
					row.append('tp')
				else:
					row.append('fp')
				writer.writerow(row)
			f.close()
			
			#Iterate all combinations to compute all tp, fp, fn, tn
			for idx1, idx2 in itertools.combinations(indices, 2):
			
				docId1 = directLookup[idx1]
				docId2 = directLookup[idx2]
			
				if (idx1, idx2) in matchedTuples: #The tuple was matched by the algorithm
					# tp or fp
					if oracle[(idx1, idx2)] == 1: #Correctly matched
						tp += 1
					else: #Incorrectly matched
						fp += 1
				else: #The tuple was not matched by the algorithm
					if oracle[(idx1, idx2)] == 1: #It should have been matched
						fn += 1
						# Add the fn to the results file
						writer.writerow([str(idx1), str(idx2), nameEmailData[idx1], nameEmailData[idx2], documents[docId1], documents[docId2], index.similarity_by_id(docId1)[docId2], 'fn'])
					else: #Did not match correctly
						tn += 1
			g.close()
			os.remove(os.path.join(dataPath, filename))
			
			#Add reflexive results to the tp's :-)
			for idx in indices:
				tp += 1
						
			try:
				precision = tp / (tp + fp)
			except:
				precision = 0.0
			
			try:
				recall = tp / (tp + fn)
			except:
				recall = 0.0
			
			try:
				fmeasure = 2 * precision * recall / (precision + recall)
			except:
				fmeasure = 0.0
					
			return [tp, fp, fn, tn, precision, recall, fmeasure]
		else:
			return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
			
	"""
		Loads the data set for each iteration, running all parameter combinations on the LSA algorithm.
		The output files for each parameter combination are augmented with the type of result (tp, fp, fn) using the function self.computeResults().
		For each iteration, a separate output file is written, containing each paramemter combination, including the precision, recall and f-measure.
		
		In: 	for 0 <= i <= nrIterations :: training_i.csv
		
		Out: 	for 0 <= j <= parameterCombination :: for 0 <= i <= nrIterations :: results_i_j.csv
				for 0 <= i <= nrIterations :: resultsTraining_i.csv
	"""
	def runTraining(self, dataPath='../data/icsmData', resultsPath='../data/icsmResults', levThrRange=[0.7, 0.8, 0.9], minLenRange=[2, 3, 4], kRange=[0.9, 0.95, 1.0], cosThrRange=[0.7, 0.75, 0.8, 0.85], nrIterations=10, nrProcesses=1):
		#Load iteration of the training set. (../data/icsmData/training_0.csv)
		#Run this iteration on all parameter combinations.
		#Compare to oracle and augment the output file from the LSA algorithm.
		#Write the results from each parameter combination, including precision, recall and f-measure. (../data/icsmResults/resultsTraining_0.csv)
		
		#Make sure the resultsPath exists. If not, create it.
		if not os.path.exists(resultsPath):
			os.makedirs(resultsPath)
		#Similar for the resultsPath/training directory, as training and testing is separated.
		if not os.path.exists(resultsPath + '/training'):
			os.makedirs(resultsPath + '/training')
		
		#Loop all iterations
		for itIdx in range(nrIterations):
			
			#Load the data
			nameEmailData = MyDict()
			f = open(os.path.join(dataPath, 'training_%d.csv' % itIdx), 'rb')
			reader = UnicodeReader(f)
			idx = 0
			for row in reader:
				try:
					alias = row[0]
					email = unspam(row[1])
					nameEmailData[idx] = (alias, email)
				except:
					print row
				idx += 1
			f.close()
			
			#Load the oracle for the data
			aliasToIdName = MyDict(os.path.join(dataPath, 'aliasToIdNameUTF8.dict'))
			oracle = computeOracle(nameEmailData, aliasToIdName)
			
			nrRuns = len(minLenRange) * len(kRange) * len(levThrRange) * len(cosThrRange)
			
			g = open(os.path.join(resultsPath, 'training', 'resultsTraining_%d.csv' % itIdx), 'wb')
			writer = csv.writer(g, delimiter=';')
			writer.writerow(['levThr','minLen','k','cosThr','tp','fp','fn','tn','precision','recall','f'])
			
			run = 0
			for levThr in levThrRange:
				for minLen in minLenRange:
					for k in kRange:
						for cosThr in cosThrRange:
							#Load the parameters
							parameters = {}
							parameters["levenshteinSimRatio"] = levThr
							parameters["minLength"] = minLen
							parameters["cosineSimRatio"] = cosThr
							parameters["rankReductionRatio"] = k
							
							#Run the LSA algorithm on these parameters
							lsaAlgo = LSAAlgo(nameEmailData, parameters, dataDir=resultsPath, dataSaveDir='training', resultsFileName='results_%d_%.2f_%d_%.2f_%.2f_preoracle.csv' % (itIdx, levThr, minLen, k, cosThr), resultsHumanReadable=False, numberOfProcesses=nrProcesses, runProfiler=False, profilerOutputDir='profilerOutput', gensimLogging=False, progressLogging=False)
							lsaAlgo.run()
							
							#Now check the results using the oracle
							[tp, fp, fn, tn, precision, recall, fmeasure] = self.computeResults(os.path.join(resultsPath, 'training'), 'results_%d_%.2f_%d_%.2f_%.2f_preoracle.csv' % (itIdx, levThr, minLen, k, cosThr), 'results_%d_%.2f_%d_%.2f_%.2f.csv' % (itIdx, levThr, minLen, k, cosThr), oracle, nameEmailData)
							
							writer.writerow([levThr,minLen,k,cosThr,tp,fp,fn,tn,precision,recall,fmeasure])
							
							run += 1
							print 'Run %d out of %d in iteration %d...' % (run, nrRuns, itIdx)
							
			g.close()

	"""
		Loads the results of the training for each iteration, grabbing the parameter combination with the best f-measure.
		This parameter combination is used for the LSA algorithm on all testing subsets.
		Similar to training, the output files are augmented with the type of result (tp, fp, fn) using the function self.computeResults().
		For each iteration, a separate output file is written, containing the results from each testing subset with the best parameter combination, including the precision, recall and f-measure.
		
		In: 	for 0 <= i <= nrIterations :: resultsTraining_i.csv
				for 0 <= i <= nrIterations :: for 0 <= j <= nrTestSubsets :: test_i_j.csv
		
		Out: 	for 0 <= i <= nrIterations :: for 0 <= j <= nrTestSubsets :: results_i_j.csv
				for 0 <= i <= nrIterations :: resultsTesting_i.csv
	"""
	def runTesting(self, dataPath='../data/icsmData', resultsPath='../data/icsmResults', nrIterations=10, nrProcesses=1):
		#For each iteration, load the file with the results for each parameter combination. (../data/icsmResults/resultsTraining_0.csv)
		#Grab the parameter combination with the highest f-measure.
		#Run the LSA algorithm on all testing sets for this iteration with the selected parameter combination. (for 0 <= i <= 9 :: ../data/icsmData/test_0_i.csv)
		#Write the results of the testing to file. (../data/icsmResults/resultsTesting_0.csv)
		
		#Make sure the resultsPath exists. If not, create it.
		if not os.path.exists(resultsPath):
			os.makedirs(resultsPath)
		#Similar for the resultsPath/testing directory, as training and testing is separated.
		if not os.path.exists(resultsPath + '/testing'):
			os.makedirs(resultsPath + '/testing')
		
		#Load the oracle for the data
		aliasToIdName = MyDict(os.path.join(dataPath, 'aliasToIdNameUTF8.dict'))
		
		#Loop all iterations
		for itIdx in range(nrIterations):
		
			resultList = []
			
			#Load the data
			f = open(os.path.join(resultsPath, 'training', 'resultsTraining_%d.csv' % itIdx), 'rb')
			reader = csv.reader(f, delimiter=';')
			for row in reader:
				header = row #skip header
				break
			for row in reader:
				#levThr;minLen;k;cosThr;tp;fp;fn;precision;recall;f
				levThr = row[0]
				minLen = row[1]
				k = row[2]
				cosThr = row[3]
				tp = row[4]
				fp = row[5]
				fn = row[6]
				tn = row[7]
				precision = row[8]
				recall = row[9]
				fMeasure = row[10]
				
				resultList.append((levThr, minLen, k, cosThr, tp, fp, fn, tn, precision, recall, fMeasure))
			f.close()
				
			#Order results by f-measure
			resultList = sorted(resultList, key=lambda tuple: -float(tuple[10]))
			
			#Grab the first record, containing the best parameters.
			bestLevThr = float(resultList[0][0])
			bestMinLen = int(resultList[0][1])
			bestK = float(resultList[0][2])
			bestCosThr = float(resultList[0][3])
			
			g = open(os.path.join(resultsPath, 'testing', 'resultsTesting_%d.csv' % itIdx), 'wb')
			writer = csv.writer(g, delimiter=';')
			writer.writerow(['levThr','minLen','k','cosThr','tp','fp','fn','tn','precision','recall','f'])
			
			parameters = {}
			parameters["levenshteinSimRatio"] = bestLevThr
			parameters["minLength"] = bestMinLen
			parameters["rankReductionRatio"] = bestK
			parameters["cosineSimRatio"] = bestCosThr
			
			#Run the LSA algorithm on all testing subsets for this iteration
			for i in range(10):
			
				#Read the data from the testing subset
				nameEmailData = MyDict()
				f = open(os.path.join(dataPath, 'test_%d_%d.csv' % (itIdx, i)), 'rb')
				reader = UnicodeReader(f)
				idx = 0
				for row in reader:
					try:
						alias = row[0]
						email = unspam(row[1])
						nameEmailData[idx] = (alias, email)
					except:
						print row
					idx += 1
				f.close()
			
				lsaAlgo = LSAAlgo(nameEmailData, parameters, dataDir=resultsPath, dataSaveDir='testing', resultsFileName='results_%d_%d_preoracle.csv' % (itIdx, i), resultsHumanReadable=False, numberOfProcesses=nrProcesses, runProfiler=False, profilerOutputDir='profilerOutput', gensimLogging=False, progressLogging=False)
				lsaAlgo.run()
				
				#Compute the oracle to verify results
				oracle = computeOracle(nameEmailData, aliasToIdName)
				
				#Now check the results using the oracle
				[tp, fp, fn, tn, precision, recall, fmeasure] = self.computeResults(os.path.join(resultsPath, 'testing'), 'results_%d_%d_preoracle.csv' % (itIdx, i), 'results_%d_%d.csv' % (itIdx, i), oracle, nameEmailData)
			
				writer.writerow([bestLevThr,bestMinLen,bestK,bestCosThr,tp,fp,fn,tn,precision,recall,fmeasure])
				
				print 'Done computing results on iteration %d, subset %d' % (itIdx, i)
				
			g.close()
			
if __name__=="__main__":

	nrIterations = 10 #out of 10
	nrProcesses = 2 #Number of CPU cores to use in the LSA algorithm
	
	levThrRange = [0.7, 0.8, 0.9]
	minLenRange = [2, 3, 4]
	kRange = [0.9, 0.95, 1.0]
	cosThrRange = [0.7, 0.75, 0.8, 0.85]

	icsmTester = ICSMTests()
	icsmTester.runTraining(levThrRange=levThrRange, minLenRange=minLenRange, kRange=kRange, cosThrRange=cosThrRange, nrIterations=nrIterations, nrProcesses=nrProcesses)
	icsmTester.runTesting(nrIterations=nrIterations, nrProcesses=nrProcesses)
			