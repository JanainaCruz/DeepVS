import numpy
from bisect import bisect
from pdb_dataset import PDBDataset


class MoleculeContextsDataset:

    def __init__(self, 
                 distanceDiscretizationIntervalSize = .3,
                 largestDistance = 5,
                 chargeDiscretizationIntervalSize = .1,
                 largestCharge  = 2.7,
                 smallestCharge = -1,
                 ):
        """
        Constructor.
        """
        self._vocabulary          = {}
        self._vocabularyByIndexes = {}
        self._dataset             = []
        self._moleculeNames       = []
        
        # constructs the intervals used to discretize the distances feature
        self._distanceDiscretizationRanges = [0]
        while self._distanceDiscretizationRanges[-1] <= largestDistance: 
            self._distanceDiscretizationRanges.append(
                    self._distanceDiscretizationRanges[-1] 
                    + distanceDiscretizationIntervalSize)
        

        # constructs the intervals used to discretize the charge feature
        self._chargeDiscretizationRanges = [smallestCharge]
        while self._chargeDiscretizationRanges[-1] <= largestCharge: 
            self._chargeDiscretizationRanges.append(
                    self._chargeDiscretizationRanges[-1] 
                    + chargeDiscretizationIntervalSize)
            

    def getVocabulary(self):
        '''
        Returns the <term, index> vocabulary.
        '''
        return self._vocabulary

    def getVocabularyByIndexes(self):
        '''
        Returns the <index, term> vocabulary.
        '''
        return self._vocabularyByIndexes

    def setVocabulary(self, vocabulary):
        '''
        Sets the vocabulary. 
        '''
        self._vocabulary = vocabulary

    def setVocabularyByIndexes(self, vocabularyByIndexes):
        '''
        Sets the vocabulary. 
        '''
        self._vocabularyByIndexes = vocabularyByIndexes

    def createExamples(self, molecules, proteinId, moleculeIdStart, 
                        moleculeIdEnd, numIntraNeighbors = 4, 
                        numInterNeighbors = 3, ligands = {}, 
                        setVocabulary=True):
        """
        For each molecule in the range [moleculeIdStart, moleculeIdEnd] creates
        a sample that contains the contexts of all atoms in the molecule.
        The context of an atom takes into consideration information from the 
        <numIntraNeighbors> closest atoms in the molecule and <numInterNeighbors>
        closest atoms in the protein.
        
        ps: An atom is considered as a neighbor of itself. Therefore, normally 
            numIntraNeighbors == numInterNeighbors + 1. 
        """
        if setVocabulary:
            self._vocabulary          = molecules.getVocabulary()
            self._vocabularyByIndexes = molecules.getVocabularyByIndexes()
            
        atomKDTrees           = molecules.getMoleculeAtomKDTrees()
        proteinAtomKDTrees    = atomKDTrees[proteinId]
        moleculeAtomPositions = molecules.getMoleculeAtomPositions()
        moleculeNames         = molecules.getMoleculeNames()
        
        for moleculeId in xrange(moleculeIdStart, moleculeIdEnd):
            molAtomKDTree    = atomKDTrees[moleculeId]
            molAtomPositions = moleculeAtomPositions[moleculeId]
            nearestNeighborAtomData = self.getNearestNeighbours(
                                    proteinAtomKDTrees, molAtomPositions,
                                        molAtomKDTree, numIntraNeighbors, 
                                            numInterNeighbors)
            
            isLigand = 0
            if moleculeNames[moleculeId] in ligands:
                isLigand = 1
                
            contexts = self.extractContextsOfMolecule(molecules, 
                                proteinId, moleculeId, nearestNeighborAtomData)
            
            self._moleculeNames.append(moleculeNames[moleculeId])
            self._dataset.append([contexts, isLigand])

    def extractContextsOfMolecule(self, molecules, proteinId, moleculeId, 
                      nearestNeighborAtomData):
        """
        For each atom in the molecule, this method creates the context features, which 
        corresponds to a vector with four elements:
            context = [
                   array with the atom type ids of inter and intra neighbor atoms,
                   array with the distances of inter and intra neighbor atoms,
                   array with the charges of inter and intra neighbor atoms,
                   array with the amino acids ids  of inter neighbor atoms,
                  ]
        """
        
        # nearestNeighborAtomData = [molNeighbourIndexes, protNeighbourIndexes,
        #            molNeighbourDistances, protNeighbourDistances]
        contexts = []
        for molNeighbourIndexes, molNeighbourDistances, \
            protNeighbourIndexes, protNeighbourDistances in nearestNeighborAtomData:
            
            atomTypes      = []
            atomCharges    = []
            atomAminoAcids = []
            # data from the molecule 
            atoms = molecules.getMoleculeAtoms()[moleculeId]
            molAtomCharges = molecules.getMoleculeAtomsCharges()[moleculeId]
#             print "molNeighbourIndexes", nearestNeighborAtomData[1][0], nearestNeighborAtomData[1][1]
#             print "protNeighbourIndexes", nearestNeighborAtomData[1][2], nearestNeighborAtomData[1][3]
            for atomIndex in molNeighbourIndexes:
                # store atom types
                atomTypes.append(atoms[atomIndex])
                # store charges
                atomCharges.append(molAtomCharges[atomIndex])

            # data from the protein 
            atoms = molecules.getMoleculeAtoms()[proteinId]
            protAtomCharges = molecules.getMoleculeAtomsCharges()[proteinId]
            protAtomAminoacids = molecules.getMoleculeAtomAminoAcides()[proteinId]
            for atomIndex in protNeighbourIndexes:
                # store atom types
                atomTypes.append(atoms[atomIndex])
                # store charges
                atomCharges.append(protAtomCharges[atomIndex])
                # store amino acids
                atomAminoAcids.append(protAtomAminoacids[atomIndex])
    
            # store distances
            atomDistances = numpy.concatenate((molNeighbourDistances, protNeighbourDistances))
    
            
            contexts.append([
                             numpy.asarray(atomTypes,numpy.int32), # atom types
                             atomDistances,  # atom distances 
                             numpy.asarray(atomCharges,numpy.float32), # charges
                             numpy.asarray(atomAminoAcids,numpy.int32) # amino acids
                             ])
        return contexts

    def getNearestNeighbours(self, proteinAtomKDTrees, moleculeAtomPositions, 
                      moleculeAtomKDTree, numIntraNeighbors, numInterNeighbors):
        """
        Retrieves for each atom in the molecule the nearest neighbor atoms:
            - intra atoms (from inside the molecule)
            - inter atoms (from the protein)
        It is important to note that each atom is included as an nearest 
        neighbor of itself.  
        """
        # stores nearest neighbor atoms in the molecule and in the protein
        nearestNeighborAtomData = []  
        
        # navigates through atom positions
        for atomPosition in moleculeAtomPositions:
            # get nearest neighbours using the KDTree query() function
            if numIntraNeighbors > 0:
                [molNeighbourDistances, molNeighbourIndexes] = \
                    moleculeAtomKDTree.query(atomPosition, k=numIntraNeighbors)
            else:
                molNeighbourDistances = numpy.zeros(0) 
                molNeighbourIndexes   = numpy.zeros(0)
            
            if numInterNeighbors > 0:
                [protNeighbourDistances, protNeighbourIndexes] = \
                    proteinAtomKDTrees.query(atomPosition, k=numInterNeighbors)
            else:
                protNeighbourDistances = numpy.zeros(0)
                protNeighbourIndexes   = numpy.zeros(0)
                
            if not isinstance(molNeighbourDistances, numpy.ndarray):
                if molNeighbourDistances == None:
                    molNeighbourDistances = numpy.zeros(0)
                else:
                    molNeighbourDistances = [molNeighbourDistances]
            if not isinstance(molNeighbourIndexes, numpy.ndarray):
                if molNeighbourIndexes == None:
                    molNeighbourIndexes = numpy.zeros(0)
                else:
                    molNeighbourIndexes = [molNeighbourIndexes]
            if not isinstance(protNeighbourDistances, numpy.ndarray):
                if protNeighbourDistances == None:
                    protNeighbourDistances = numpy.zeros(0)
                else:
                    protNeighbourDistances = [protNeighbourDistances]
            if not isinstance(protNeighbourIndexes, numpy.ndarray):
                if protNeighbourIndexes == None:
                    protNeighbourIndexes = numpy.zeros(0)
                else:
                    protNeighbourIndexes = [protNeighbourIndexes]
            
            nearestNeighborAtomData.append([
                    molNeighbourIndexes, molNeighbourDistances,
                    protNeighbourIndexes, protNeighbourDistances])

        return nearestNeighborAtomData

    def getTermByIndex(self, termIndex):
        '''
        Returns the feature index. 
        '''
        return self._vocabularyByIndexes.get(termIndex)
    
    def getTermIndexAdd(self, term):
        '''
        Returns the term index.
        If it does not exists, a new entry is created in the vocabulary.    
        '''
        termIndex = self._vocabulary.get(term)
        if termIndex:
            return termIndex
        else:
            termIndex = len(self._vocabulary) + 1
            self._vocabulary[term] = termIndex
            self._vocabularyByIndexes[termIndex] = term 
            return termIndex

    def getTermIndex(self, term):
        '''
        Returns the word index.
        '''
        termIndex = self._vocabulary.get(term)
        if termIndex == None:
            raise Exception("Term '%s' does not exist in the vocabulary."%term)
        return termIndex

    def discretizeDistance(self):
        '''
        Discretizes the distances feature .
        '''
        
        # get the index in the vocabulary of each discretization interval
        vocIndexOfDiscretizationRanges = []
        for intervalBorder in self._distanceDiscretizationRanges:  
            vocIndexOfDiscretizationRanges.append(
                                    self.getTermIndexAdd("%.1f"%intervalBorder))
        
        distancesFtrIndex = 1
        for moleculeData in self._dataset:
            for contextData in moleculeData[0]:
                discretizedValues = numpy.zeros(contextData[distancesFtrIndex].shape, 
                                                dtype=numpy.int32)
                i = 0
                for distance in contextData[distancesFtrIndex]:
                    # Gets the vocabulary index of the range where the distance lies in.
                    # Bisect() is a function that finds the (interval) position in 
                    #  self.__distanceDiscretizationRanges where the distance
                    # should be put in
                    discretizedValues[i] =  vocIndexOfDiscretizationRanges[
                                                bisect(self._distanceDiscretizationRanges, distance) -1 ]
                    i += 1
                contextData[distancesFtrIndex] = discretizedValues
                           

    def discretizeCharges(self):
        '''
        Discretizes the charges feature .
        '''
        # get the index in the vocabulary of each discretization interval
        vocIndexOfDiscretizationRanges = []
        for intervalBorder in self._chargeDiscretizationRanges:  
            vocIndexOfDiscretizationRanges.append(
                                    self.getTermIndexAdd("%.1f"%intervalBorder))

        chargesFtrIndex = 2
        for moleculeData in self._dataset:
            for contextData in moleculeData[0]:
                discretizedValues = numpy.zeros(contextData[chargesFtrIndex].shape, 
                                                dtype=numpy.int32)
                i = 0
                for charge in contextData[chargesFtrIndex]:
                    discretizedValues[i] =  vocIndexOfDiscretizationRanges[
                                                bisect(self._chargeDiscretizationRanges, charge) -1 ]
                    i += 1
                contextData[chargesFtrIndex] = discretizedValues

    def saveToFile(self, fileName):
        '''
        Saves the loaded context dataset to a text file.
        '''
        f = open(fileName, 'w')
        # for each molecule
        for molData, molName in zip(self._dataset, self._moleculeNames):
            f.write("@%s,%d\n"%(molName, molData[1]))
            # for each molecule's context
            for contextData in molData[0]:
                ftrValues = []
                # for each of the four basic features: atomtype, distance, charge, aminoacid (in this order)
                for ftrValuesOfOneFeature in contextData:
                    ftrValues.extend([self.getTermByIndex(x) for x in ftrValuesOfOneFeature])
                f.write("%s\n"%(','.join(ftrValues)))
            f.write("\n")
        f.close()

    def getFeatureAsDictionary(self, featureName):
        '''
        Creates and returns a dictionary that contains the values of the given 
        feature.
        '''
        tempDict   = {}
        featureIndex = self.getFeatureIndex(featureName)
        for moleculeData in self._dataset:
            for contextData in moleculeData[0]:
                for ftrValue in contextData[featureIndex]:
                    if not ftrValue in tempDict:
                        tempDict[ftrValue] = True
        
        resultDict = {}
        valueId    = 0
        for ftrValue in tempDict.iterkeys():
            resultDict[self.getTermByIndex(ftrValue)] = valueId
            valueId += 1

        return resultDict

    def getNumberOfMolecules(self):
        '''
        Returns the number of molecules in the dataset. 
        '''
        return len(self._dataset)

    def sortByMoleculeSize(self, reverse=True):
        """
        Sorts the dataset by molecule size.
        """
        self._dataset = sorted(self._dataset, key=len, reverse = reverse)
        