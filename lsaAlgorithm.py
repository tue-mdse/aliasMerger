"""
	Copyright 2012-2013
	Eindhoven University of Technology (Erik Kouters, Bogdan Vasilescu, Alexander Serebrenik)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from gensim import models, similarities, matutils, corpora, utils
from scipy.sparse import coo_matrix
#from sparsesvd import sparsesvd
from dictUtils import MyDict
import itertools
from filters import *
import csv
from time import clock
#from Levenshtein import *
import Levenshtein #ez_setup.py python-Levenshtein
#import editdist #ez_setup.py py-editdist
#from unicodeMagic import UnicodeReader, UnicodeWriter
from stringcmp import do_stringcmp
import math
import numpy
from multiprocessing import Process, Queue, Lock, current_process
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double
import time, random
import cProfile
import gc
import os
from unicodeMagic import UnicodeReader, UnicodeWriter
#from scipy.spatial.distance import pdist
#from scipy.spatial import distance
#from numpy.linalg import inv
	
class LSAAlgo():
	"""
	LSA-based implementation of the alias merging algorithm.
	
	Kouters E, Vasilescu B, Serebrenik A and van den Brand, MGJ (2012), 
	Who's who in GNOME: using LSA to merge software repository identities, 
	In Proc. 28th IEEE International Conference on Software Maintenance, pp 592-595.
	"""
	
	def __init__(self, data, parameters, dataDir, dataSaveDir, resultsFileName='results.csv', resultsHumanReadable=True, numberOfProcesses=1, runProfiler=False, profilerOutputDir='profilerOutput', gensimLogging=False, progressLogging=True):
		"""
		Constructor:
		- data is a MyDict object, key = integer id, value = (name, email) tuple.
		- parameters is a dictionary of parameters. To be interpreted by the run function.
		- whereToStore is a path to a csv file to put results in.
		"""
		self.nameEmailData = data
		self.parameters = parameters
		self.dataDir = dataDir
		self.dataSaveDir = dataSaveDir
		self.emailExceptions = set(["jira@apache.org"])
		self.numberOfProcesses = numberOfProcesses
		self.runProfiler = runProfiler
		self.profilerOutputDir = profilerOutputDir
		
		self.documentsFileName = 'documents'
		self.directLookupFileName = 'directLookup'
		self.reverseLookupFileName = 'reverseLookup'
		self.dictionaryFileName = 'dictionary'
		self.originalCorpusFileName = 'corpus'
		self.termCorpusFileName = 'termCorpus'
		self.levenshteinCorpusFileName = 'levCorpus'
		self.tfidfModelFileName = 'tfidf'
		self.lsiModelFileName = 'lsi'
		self.similarityIndexFileName = 'index'
		
		self.dataProcessingDir = 'processing'
		
		self.resultsFileName = resultsFileName
		self.resultsHumanReadable = resultsHumanReadable
		
		if gensimLogging:
			import logging
			logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		
		self.progressLogging = progressLogging
		
	def worker(self, queue, core_nr, dictionary, levThr):
	
		if self.runProfiler:
			cProfile.runctx('self.handleWorker(queue, core_nr, dictionary, levThr)', globals(), locals(), '%s/%s/computeTermCorpus_%i' % (self.dataDir, self.profilerOutputDir, core_nr))
		else:
			self.handleWorker(queue, core_nr, dictionary, levThr)
		
	def handleWorker(self, queue, core_nr, dictionary, levThr):

		corpus_writer = matutils.MmWriter('%s/%s/%s_%i.mm' % (self.dataDir, self.dataSaveDir, self.termCorpusFileName, core_nr))
		corpus_writer.write_headers(-1, -1, -1)
		nnz = 0
		nrDocs = 0
		
		poslast = 0
		offsets = []
		
		# dictionary lookup is very slow. Copy to own version.
		mydictionary = {}
		for key in dictionary.iterkeys():
			mydictionary[key] = (dictionary[key], len(dictionary[key]))
		dictionaryLen = len(dictionary)
		
		for item in iter(queue.get, 'STOP'):
			termIndex = item
			
			# Determine offsets for the index file, allowing O(1) access time of documents.
			posnow = corpus_writer.fout.tell()
			if posnow == poslast:
				offsets[-1] = -1
			offsets.append(posnow)
			poslast = posnow
			
			nnz += self.writeLevenshteinDistance(termIndex, mydictionary, dictionaryLen, levThr, corpus_writer)
			nrDocs += 1
		corpus_writer.fake_headers(nrDocs, dictionaryLen, nnz)
		corpus_writer.close()
		
		# Write index to file
		index_fname = corpus_writer.fname + '.index'
		utils.pickle(offsets, index_fname)

	def writeLevenshteinDistance(self, termIndex, dictionary, dictionaryLen, levThr, corpus_writer):

		term, termLen = dictionary[termIndex]
		#termLen = len(term)

		# Compute candidates for levenshtein distance of term from termToRowIndex.
		#candidates = []
		candidates = [None] * dictionaryLen
		i = 0
		for candidateId, (candidate, candidateLen) in dictionary.iteritems():
			#candidate, candidateLen = dictionary[candidateId]
			
			# Add candidate if the difference between termLen and candidateLen is less or equal than 0.5*maxlen.
			if termLen >= candidateLen:
				diff = termLen - candidateLen
				if diff <= 0.5*termLen:
					candidates[i] = (candidateId, candidate, candidateLen)
					i += 1
			else:
				diff = candidateLen - termLen
				if diff <= 0.5*candidateLen:
					candidates[i] = (candidateId, candidate, candidateLen)
					i += 1
		
		#Grab the sublist excluding the preallocated values
		candidates = candidates[0:i]
		
			#if 2*candidateLen >= termLen:
			#	candidates.append((candidateId, candidate, candidateLen))
			#elif 2*termLen >= candidateLen:
			#	candidates.append((candidateId, candidate, candidateLen))
		#candidates = [(candidateId, dictionary[candidateId]) for candidateId in dictionary.keys() if 2*len(dictionary[candidateId]) >= termLen]
		#candidates += [(candidateId, dictionary[candidateId]) for candidateId in dictionary.keys() if 2*termLen >= len(dictionary[candidateId])]
		#candidates = set(candidates)
		
		# Compute the values
		#sims = []
		sims = [None] * dictionaryLen
		i = 0
		for candidateId, candidate, candidateLen in candidates:
			sim = 0.0
			#Compute distance between candidate and term
			if termLen >= candidateLen: #Split on len to prevent the use of max, which is awfully slow.
				sim = 1.0 - float(Levenshtein.distance(candidate, term))/termLen #Compute Levensthein distance
			else:
				sim = 1.0 - float(Levenshtein.distance(candidate, term))/candidateLen #Compute Levensthein distance
			if sim >= levThr:
				sims[i] = (candidateId, sim)
				i += 1
					
		#Grab the sublist excluding the preallocated values
		sims = sims[0:i]
		
		max_id, veclen = corpus_writer.write_vector(termIndex, sims)
		return veclen
		
	def precomputeLevenshteinDistanceBetweenAllTerms(self, numTerms, dictionary, levThr):
		TASKS = [termIndex for termIndex in range(numTerms)]
		# Create queues
		task_queue = Queue()
		# Submit tasks
		for task in TASKS:
			task_queue.put(task)
		tasks = []
		for i in range(self.numberOfProcesses):
			p = Process(target=self.worker, args=(task_queue, i, dictionary, levThr))
			p.start()
			tasks.append(p)
		# Write 'STOP' to the queue for numberOfProcesses times to have each process spawned quit on one 'STOP'.
		for i in range(self.numberOfProcesses):
			task_queue.put('STOP')
		for task in tasks: #Wait for the tasks to finish
			task.join()
		
	"""
	The Levenshtein distance between each term has been precomputed in parallel.
	Each process saved the results in separate files.
	This function merges these separate files into a single corpus.
	
	termCorpusFileName : The name of the termCorpus file (e.g. termCorpus_0.mm, termCorpus_1.mm => termCorpus.mm)
	"""
	def mergeIntoTermCorpus(self):
		corpus_writer = matutils.MmWriter('%s/%s/%s.mm' % (self.dataDir, self.dataSaveDir, self.termCorpusFileName))
		corpus_writer.write_headers(-1, -1, -1)
		
		num_nnz = 0
		poslast = 0
		offsets = []
		
		write_index = 0
		totalLen = 0
		corporaDict = MyDict()
		for i in range(self.numberOfProcesses):
			corpus = corpora.MmCorpus('%s/%s/%s_%i.mm' % (self.dataDir, self.dataSaveDir, self.termCorpusFileName, i))
			#corporaList.append(corpus)
			# (current termId, current index in corpus, corpus)
			if len(corpus) > 0:
				termId = [id for (id, sim) in corpus[0] if sim == 1.0][0]
				corporaDict[i] = (termId, 0, corpus, len(corpus))
				totalLen += len(corpus)
			
		while 1:
			isDone = False
			for corpusId in corporaDict.keys():
				termId, index, corpus, len_corpus = corporaDict[corpusId] # Read all values for current corpus from MyDict.
				if termId == write_index: # We are writing to the merged corpus at index 'write_index'. Write it if it coincides with the column id of the current corpus.
				
					# Determine offsets for the index file, allowing O(1) access time of documents.
					posnow = corpus_writer.fout.tell()
					if posnow == poslast:
						offsets[-1] = -1
					offsets.append(posnow)
					poslast = posnow
				
					# Write current document
					max_id, veclen = corpus_writer.write_vector(write_index, corpus[index])
					num_nnz += veclen
					
					# Update values
					write_index += 1 #Update the write index of the merged corpus
					index += 1 #Update the index of the current corpus
					if index == len_corpus: #Reached the end of the current corpus. Set values to -1 so no more document will be grabbed from this corpus.
						corporaDict[corpusId] = (-1, -1, corpus, len_corpus) #Set index to -1. Corpus has been fully read.
					else:
						termId = [id for (id, sim) in corpus[index] if sim == 1.0][0] #Grab the next column id :: TODO -- CAN THIS BE DONE MORE EFFICIENTLY?
						corporaDict[corpusId] = (termId, index, corpus, len_corpus) #Update the MyDict with the new values of the current corpus
					
					if write_index == totalLen: # If all corpora have been fully read, exit the while loop.
						isDone = True
						
			if isDone:
				break
		corpus_writer.fake_headers(totalLen, totalLen, num_nnz)
		corpus_writer.close()
		
		# Write index to file
		index_fname = corpus_writer.fname + '.index'
		utils.pickle(offsets, index_fname)
		
	"""
	Uses the original corpus and the precomputed termCorpus to create the corpus which includes the levenshtein distance 
	"""
	def createLevenshteinCorpus(self):
		corpus = corpora.MmCorpus('%s/%s/%s.mm' % (self.dataDir, self.dataSaveDir, self.originalCorpusFileName))
		termCorpus = corpora.MmCorpus('%s/%s/%s.mm' % (self.dataDir, self.dataSaveDir, self.termCorpusFileName))
		
		levCorpus_writer = matutils.MmWriter('%s/%s/%s.mm' % (self.dataDir, self.dataSaveDir, self.levenshteinCorpusFileName))
		levCorpus_writer.write_headers(-1, -1, -1)
		num_nnz = 0
		poslast = 0
		offsets = []
		
		write_index = 0
		
		for doc in corpus:
			doc_ids = [id for (id, term) in doc]
			
			candidates = []
			levDoc = []
			
			for termId in doc_ids:
				# Create a list of candidates for this document
				for candidate in termCorpus[termId]:
					candidates.append(candidate)
					
			candidates = sorted(candidates, key=lambda tuple: tuple[0])
					
			it = itertools.groupby(candidates, key=lambda tuple: tuple[0]) #Group together similarity measures for the same term.
			for key, subiter in it:
				levDoc.append(max(subiter, key=lambda tuple: tuple[1])) #From the multiple similarity measures, grab the maximum.
			
			# Determine offsets for the index file, allowing O(1) access time of documents.
			posnow = levCorpus_writer.fout.tell()
			if posnow == poslast:
				offsets[-1] = -1
			offsets.append(posnow)
			poslast = posnow
		
			#Write the document to levCorpus in file
			max_id, veclen = levCorpus_writer.write_vector(write_index, levDoc)
			num_nnz += veclen
			
			write_index += 1
		
		levCorpus_writer.fake_headers(len(corpus), len(termCorpus), num_nnz)
		levCorpus_writer.close()
		
		# Write index to file
		index_fname = levCorpus_writer.fname + '.index'
		utils.pickle(offsets, index_fname)
		
	def writeResults(self, index, reverseLookup, documents, simThr):
		f = open('%s/%s/%s' % (self.dataDir, self.dataSaveDir, self.resultsFileName), 'wb')
		writer = csv.writer(f, delimiter=";")

		""" So far unimplemented.
		if self.resultsHumanReadable:
			for docIdx1, simi in enumerate(index):
				aliases1 = reverseLookup[docIdx1]
				for docIdx2, val in enumerate(simi):
					if docIdx1 > docIdx2:
						if val >= simThr:
							aliases2 = reverseLookup[docIdx2]
							for id1 in aliases1:
								for id2 in aliases2:
									writer.writerow([str(id1)]+list(self.nameEmailData[id1])+[str(id2)]+list(self.nameEmailData[id2])+[str(val)])
		else:		
		"""
		writer.writerow(['leftid', 'rightid', 'left','right','textLeft','textRight','sim'])
		for docIdx1, simi in enumerate(index):
			aliases1 = reverseLookup[docIdx1]
			doc1 = documents[docIdx1]
			for docIdx2, val in enumerate(simi):
				if docIdx1 > docIdx2:
					if val >= simThr:
						aliases2 = reverseLookup[docIdx2]
						doc2 = documents[docIdx2]
						for id1 in aliases1:
							tuple1 = self.nameEmailData[id1]
							for id2 in aliases2:
								tuple2 = self.nameEmailData[id2]
								#writer.writerow([str(id1)]+list(self.nameEmailData[id1])+[str(id2)]+list(self.nameEmailData[id2])+[str(val)])
								writer.writerow([str(id1), str(id2), tuple1, tuple2, doc1, doc2, str(val)])
								
			#Next to the combinations between the documents, we must not forget about the aliases that are within the same document.
			#These aliases have been merged into the same document (e.g. by having an equal email address)
			#Therefore these aliases have the same document string, leading to a similarity of 1.0.
			for id1, id2 in itertools.combinations(aliases1, 2):
				tuple1 = self.nameEmailData[id1]
				tuple2 = self.nameEmailData[id2]
				writer.writerow([str(id1), str(id2), tuple1, tuple2, doc1, doc1, str(1.0)])
					
										
		f.close()
		
	def run(self):
		## Decode parameters
		levThr  = self.parameters["levenshteinSimRatio"] # threshold for Levenshtein similarity
		minLen  = self.parameters["minLength"] # threshold for min length of a term
		simThr	= self.parameters["cosineSimRatio"] # threshold for cosine similarity between documents
		k		= self.parameters["rankReductionRatio"] # dimensionality reduction ratio (0 < k <= 1)
		
#		dataIndices = self.nameEmailData.keys()
		
		"""I will iteratively build and populate the document-term matrix."""
		runningColIdx = 0
		runningRowIdx = 0
		
		## 
		reverseLookup = MyDict()
		directLookup = MyDict()
		
		documents = MyDict() # Dictionary of documents (key = column index)
		terms = MyDict() # Dictionary of terms (key = row index)
		allTerms = set() # Set of all terms
		
		termToColumnIndex = MyDict()
		termToRowIndex = MyDict()
		
		start = clock()
		emailToColumnIndex = MyDict()
		for idx in self.nameEmailData.keys():
			addEmailPrefix = True # default value (always add email prefix to document string, unless ..)
			
			(name, email) = self.nameEmailData[idx] # the input data contains (name,email) tuples
			#Easy merge: different names used by the same email address.
			#Exceptions: mailing list email addresses.
			if email not in self.emailExceptions:
				#If email address is personal, then:
				#- If I have seen it before, retrieve the column (document) to which it corresponds.
				#- If I see it for the first time, assign it a new column (document).
				try:
					columnIdx = emailToColumnIndex[email]
				except:
					columnIdx = runningColIdx
					emailToColumnIndex[email] = columnIdx
					runningColIdx += 1
			else:
				#If email address is not personal, directly assign it a new column (document).
				columnIdx = runningColIdx
				runningColIdx += 1
				addEmailPrefix = False # remember not to add email prefix to document string
				
			#Create lookup tables to move from input data to doc-term matrix and back.
			directLookup[idx] = columnIdx # from index in the input data to index of document (column) in doc-term matrix
			reverseLookup.append(columnIdx, idx) # from document index (in doc-term matrix) to index in the input data
			
			#Add name parts to document string if long enough and not consisting of digits, after normalization.
			docString = [namePart for namePart in normalize(name) if len(namePart) >= minLen and not namePart.isdigit()]
			
			if addEmailPrefix:
				#Add whole email address prefix to document string, if long enough and after after normalization.
				prefix = email.split('@')[0].split('+')[0]
				docString += [prefixPart for prefixPart in normalize(prefix) if len(prefixPart) >= minLen and not prefixPart.isdigit()]
				#Split email address prefix on dot, and add each part if long enough and after normalization.
				prefixParts = prefix.split('.')
				for prefixPart in prefixParts:
					docString += [p for p in normalize(prefixPart) if len(p) >= minLen and not p.isdigit()]
			
			#The document string is a set of terms. Figure out to which rows in the doc-term matrix the terms correspond.
			for term in docString:
				try:
					#If I've seen the term before, retrieve row index.
					rowIdx = termToRowIndex[term]
				except:
					#If first time I see the term, assign new row.
					rowIdx = runningRowIdx
					termToRowIndex[term] = rowIdx
					runningRowIdx += 1
				#Keep track of which term is at which row index.
				terms[rowIdx] = term
			
			#Update the document at current column index with new document string.
			try:
				documents[columnIdx].update(docString)
			except:
				documents[columnIdx] = set() # if here then it's the first occurrence of this email address (document)
				documents[columnIdx].update(docString)
				
			#Keep an updated set of terms (from all documents).
			allTerms.update(docString)
			
			#Bookkeeping: keep track of the documents in which a term appears.
			for term in set(docString):
				try:
					termToColumnIndex[term].add(columnIdx)
				except:
					termToColumnIndex[term] = set([columnIdx])
		end = clock()
				
		numDocuments = len(documents.keys())
		numTerms = len(allTerms)
		if self.progressLogging:
			print "Initial pass: %6.3f seconds (%d terms, %d documents)" % ((end - start), numTerms, numDocuments)
		
		#Make sure that dataDir/saveDataDir exists.
		if not os.path.exists('%s/%s' % (self.dataDir, self.dataSaveDir)):
			os.makedirs('%s/%s' % (self.dataDir, self.dataSaveDir))
		#Make sure that dataDir/saveDataDir/dataProcessingDir exists.
		if not os.path.exists('%s/%s/%s' % (self.dataDir, self.dataSaveDir, self.dataProcessingDir)):
			os.makedirs('%s/%s/%s' % (self.dataDir, self.dataSaveDir, self.dataProcessingDir))
			
		if self.runProfiler:
			#Make sure that dataDir/saveDataDir/profilerOutputDir exists.
			if not os.path.exists('%s/%s/%s' % (self.dataDir, self.dataSaveDir, self.profilerOutputDir)):
				os.makedirs('%s/%s/%s' % (self.dataDir, self.dataSaveDir, self.profilerOutputDir))
				
		#Save the documents MyDict and directLookup MyDict for later reference
		documents.save('%s/%s/%s.dict' % (self.dataDir, self.dataSaveDir, self.documentsFileName))
		directLookup.save('%s/%s/%s.dict' % (self.dataDir, self.dataSaveDir, self.directLookupFileName))
		reverseLookup.save('%s/%s/%s.dict' % (self.dataDir, self.dataSaveDir, self.reverseLookupFileName))
		
		#Build the dictionary and corpus on disk from the documents.
		texts = [documents[key] for key in documents.keys()]
		dictionary = corpora.Dictionary(texts) # Dictionary: {'nalley': 5649, 'francesco': 4634, 'caisse': 3381, 'gabrielle': 1097, ...} :: (term: id)
		dictionary.save('%s/%s/%s.dict' % (self.dataDir, self.dataSaveDir, self.dictionaryFileName)) # Save dictionary
		corpus = [dictionary.doc2bow(text) for text in texts] # Corpus: [[(0, 1), (1, 1), (2, 1)], [(3, 1), (4, 1)], [(5, 1), (6, 1)], ...] :: (id, count)
		corpora.MmCorpus.serialize('%s/%s/%s.mm' % (self.dataDir, self.dataSaveDir, self.originalCorpusFileName), corpus)
		
		# Precompute levenshtein distances between all terms on all cpu cores
		if self.progressLogging:
			print 'Starting computation of levenshtein...'
		start = clock()
		if self.runProfiler:
			cProfile.runctx('self.precomputeLevenshteinDistanceBetweenAllTerms(numTerms, dictionary, levThr)', globals(), locals(), '%s/%s/%s/computeTermCorpusChunks' % (self.dataDir, self.dataSaveDir, self.profilerOutputDir))
		else:
			self.precomputeLevenshteinDistanceBetweenAllTerms(numTerms, dictionary, levThr)
		end = clock()
		if self.progressLogging:
			print 'Done computing levenshtein: %6.3f seconds' % (end - start)
		
		# Merge separate MmCorpus files
		start = clock()
		if self.runProfiler:
			cProfile.runctx('self.mergeIntoTermCorpus()', globals(), locals(), '%s/%s/%s/mergeIntoTermCorpus' % (self.dataDir, self.dataSaveDir, self.profilerOutputDir))
		else:
			self.mergeIntoTermCorpus()
		end = clock()
		if self.progressLogging:
			print 'Done merging corpora to termCorpus: %6.3f seconds' % (end - start)
		
		# Create levCorpus from corpus.mm and termCorpus.mm.
		start = clock()
		if self.runProfiler:
			cProfile.runctx('self.createLevenshteinCorpus()', globals(), locals(), '%s/%s/%s/createLevenshteinCorpus' % (self.dataDir, self.dataSaveDir, self.profilerOutputDir))
		else:
			self.createLevenshteinCorpus()
		end = clock()
		if self.progressLogging:
			print 'Done creating levCorpus from corpus and termCorpus: %6.3f seconds' % (end - start)
		
		corpus_disk = corpora.MmCorpus('%s/%s/%s.mm' % (self.dataDir, self.dataSaveDir, self.levenshteinCorpusFileName))
		
		""" Having the document-term matrix (including Levenshtein distance) in the corpus, we apply the TFIDF model. """
		start = clock()
		tfidf = models.TfidfModel(corpus_disk)
		tfidf.save('%s/%s/%s' % (self.dataDir, self.dataSaveDir, self.tfidfModelFileName))
		corpus_tfidf = tfidf[corpus_disk]
		end = clock()
		if self.progressLogging:
			print "Inverse document frequency: %6.3f seconds" % (end - start)
		
		""" Next up is applying the LSI model on top of the TFIDF model. """
		start = clock()
		number_topics = len(corpus_disk)*k
		try:
			#Occasionally computing the SVD fails.
			#http://projects.scipy.org/numpy/ticket/990
			lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=number_topics)
		except:
			print 'Failed to compute the LSI model. (SVD did not converge?)'
			return
		lsi.save('%s/%s/%s' % (self.dataDir, self.dataSaveDir, self.lsiModelFileName))
		corpus_lsi = lsi[corpus_tfidf]
		end = clock()
		if self.progressLogging:
			print "LSI model: %6.3f seconds" % (end - start)
		
		""" Finally, we run the cosine similarity on the matrix. """
		start = clock()
		index = similarities.docsim.Similarity('%s/%s/%s/%s' % (self.dataDir, self.dataSaveDir, self.dataProcessingDir, self.similarityIndexFileName), corpus_lsi, number_topics)
		index.save('%s/%s/%s' % (self.dataDir, self.dataSaveDir, self.similarityIndexFileName))
		end = clock()
		if self.progressLogging:
			print "Similarities index: %6.3f seconds" % (end - start)
		
		"""Output results."""
		start = clock()
		if self.runProfiler:
			cProfile.runctx('self.writeResults(index, reverseLookup, documents, simThr)', globals(), locals(), '%s/%s/%s/createLevenshteinCorpus' % (self.dataDir, self.dataSaveDir, self.profilerOutputDir))
		else:
			self.writeResults(index, reverseLookup, documents, simThr)
		end = clock()
		if self.progressLogging:
			print "Writing results: %6.3f seconds" % (end - start)
				
		return
		
def decodeHeader(s):
	headers = email.header.decode_header(s)
	decodedString = ''
	for header in headers:
	
		headerStr = header[0]
		charset = header[1]
		if headerStr.endswith('\\'):
			headerStr = headerStr.rstrip('\\')
			newHeader = email.header.decode_header(headerStr)[0]
			headerStr = newHeader[0]
			charset = newHeader[1]
		if headerStr.find('\n') != -1:
			headerStr = headerStr.replace('\n', '')
			newHeader = email.header.decode_header(headerStr)[0]
			headerStr = newHeader[0]
			charset = newHeader[1]
		
		if charset == 'x-unknown':
			charset = 'iso-8859-1'
		elif charset == 'iso-8859-8-i':
			charset = 'iso-8859-8'
		elif charset == 'baltic':
			charset = 'iso8859-4'
		elif charset == 'x-windows-949':
			charset = 'cp949'
		elif charset == 'x-gbk':
			charset = 'gbk'
		elif charset == 'gb2312':
			charset = 'gbk'
		elif charset == 'x-mac-cyrillic':
			charset = 'maccyrillic'
		elif charset == '\xfdso-8859-9':
			charset = 'iso-8859-9'
		elif charset is None:
			charset = 'unicode-escape' #Fall back on unicode-escape when charset is none
			
		decoded = False
		attempt = 0
		while not decoded:
			try:
				attempt += 1
				decodedString += ' ' + headerStr.decode(charset)
				decoded = True
			except:
				#print attempt, header, charset
				if attempt == 1:
					charset = 'iso-8859-1'
				elif attempt == 2:
					charset = 'iso-8859-2'
				elif attempt == 3:
					charset = 'utf-8'
				elif attempt == 4:
					charset = 'unicode-escape'
				else:
					print 'stopping...'
					exit()
		
	return decodedString.lstrip()
		
if __name__=="__main__":
	import os
	from unicodeMagic import UnicodeReader, UnicodeWriter
	
	dataPath = os.path.relpath("../data/")
	
#	f = open(os.path.join(dataPath, "testData.csv"), "rb")
#	reader = UnicodeReader(f)	
#	g = open(os.path.join(dataPath, "testData2.csv"), "wb")
#	writer = UnicodeWriter(g)
#	idx = 1
#	for row in reader:
#		writer.writerow([str(idx)] + row)
#		idx += 1
#	g.close()
#	exit()

	# Choose dataset
	#dataset = 'aliases2'
	dataset = 'gnome'
	dataset = 'icsm'
	
	# Choose dataset size
	datasetsize = 'full'
	#datasetsize = 'sample'
	
	nameEmailData = MyDict()
	if dataset == 'aliases2':
		f = open(os.path.join(dataPath, "aliases2.csv"), "rb")
	#	f = open(os.path.join(dataPath, "testData2.csv"), "rb")
		reader = UnicodeReader(f)
		header = reader.next()
			
		for row in reader:
			try:
				idx = int(row[0])
				name = row[1]
				email = unspam(row[2])
				nameEmailData[idx] = (name, email)
			except:
				print row
		f.close()
		print 'Using the aliases2.csv data set...'
	elif dataset == 'gnome':
		import email.utils
		import email.header
	
		emailAddressToUniqueNames = MyDict(os.path.join(dataPath,'emailAddressToUniqueNamesBlacklisted.dict'))
		i = 0
		emailAddresses = [emailAddress for emailAddress in emailAddressToUniqueNames.keys() if len(emailAddress) > 1]
		for emailAddress in emailAddresses:
			#print emailAddressToUniqueNames[emailAddress]
			decodedNames = [decodeHeader(uniqueName) for uniqueName in emailAddressToUniqueNames[emailAddress] ]
			decodedNames = [decodedName for decodedName in decodedNames if len(decodedName) > 1]
			
			for name in decodedNames:
				nameEmailData[i] = (name, emailAddress)
				i += 1
		print 'Using the GNOME Mailing List data set...'
	elif dataset == 'icsm':
		itIdx = 8
		f = open(os.path.join(dataPath, 'icsmData', 'training_%d.csv' % itIdx), 'rb')
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
		print 'Using the ICSM`12 ERA data set...'

	data = MyDict()
	if datasetsize == 'sample':
		import random
		sampleIdx = random.sample(nameEmailData.keys(), len(nameEmailData.keys())/40)
		#print len(sampleIdx), 'rows in sample'
		sample = MyDict()
		for idx in sampleIdx:
			sample[idx] = nameEmailData[idx]
		data = sample
		print 'Grabbed a sample of the data set...'
	elif datasetsize == 'full':
		data = nameEmailData
		print 'Using the full data set...'
	
	numRows = len(data.keys())
	print numRows, 'rows'
		
	minLenRange = [2,3,4]
	minLen = 2
	
	k = 0.50
	
	levThrRange = [0.5,0.75]
	levThr = 0.5
				
	thresholds = [0.65,0.7,0.75]
	cosThr = 0.7
	
	parameters = {}
	parameters["levenshteinSimRatio"] = levThr
	parameters["minLength"] = minLen
	parameters["cosineSimRatio"] = cosThr
	parameters["rankReductionRatio"] = k
	
	start = clock()
	
	dataSaveDir = dataset + datasetsize # {aliases2/gnome} + {sample/full} (e.g. gnomefull)
	
	lsaAlgo = LSAAlgo(data, parameters, dataDir='../data', dataSaveDir=dataSaveDir, resultsFileName='results.csv', resultsHumanReadable=True, numberOfProcesses=2, runProfiler=False, profilerOutputDir='profilerOutput', gensimLogging=False)
	lsaAlgo.run()
	end = clock()
	print "== Total time: %6.3f seconds" % (end - start)
	
