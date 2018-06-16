#! -*- coding: UTF-8 -*-
from codecs import open

import os

import numpy
from scipy import spatial

from re import compile, subn


class PDBDataset:
    """
    This class implements a dataset that can handle PDB/MOL2 files.
    """

    def __init__(self, vocabulary = {}, vocabularyByIndexes = {}):
        """
        Constructor.
        """
        self._vocabulary          = vocabulary
        self._vocabularyByIndexes = vocabularyByIndexes

        self._moleculeNames           = []
        self._moleculeAtoms           = []
        self._moleculeAtomNames       = []
        self._moleculeAtomPositions   = []
        self._moleculeAtomAminoAcides = []
        self._moleculeAtomCharges     = []
        self._moleculeAtomBonds       = []
        
        self._moleculeAtomKDTrees = []

        self.__number = compile("[0-9']")

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

    def getVocabulary(self):
        '''
        Returns the vocabulary. 
        '''
        return self._vocabulary
    
    def getVocabularyByIndexes(self):
        '''
        Returns the vocabularyByIndexes. 
        '''
        return self._vocabularyByIndexes
    
    def getMoleculeNames(self):
        '''
        Returns self._moleculeNames. 
        '''
        return self._moleculeNames

    def getMoleculeAtomPositions(self):
        '''
        Returns self._moleculeAtomPositions. 
        '''
        return self._moleculeAtomPositions

    def getMoleculeAtomKDTrees(self):
        '''
        Returns self._moleculeAtomKDTrees. 
        '''
        return self._moleculeAtomKDTrees

    def getMoleculeAtoms(self):
        '''
        Returns self._moleculeAtoms. 
        '''
        return self._moleculeAtoms

    def getMoleculeAtomsCharges(self):
        '''
        Returns self._moleculeAtomCharges. 
        '''
        return self._moleculeAtomCharges

    def getMoleculeAtomAminoAcides(self):
        '''
        Returns self._moleculeAtomAminoAcides. 
        '''
        return self._moleculeAtomAminoAcides

    def getNumberOfAtoms(self):
        '''
        Returns the total number of atoms. 
        '''
        s  = 0
        for segment in self._moleculeAtoms:
            s += len(segment)
        
        return s

    def getNumberOfMolecules(self):
        '''
        Returns the number of molecules. 
        '''
        return len(self._moleculeNames)
    
    def getTermByIndex(self, termIndex):
        '''
        Returns the term given the index. 
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

    def cleanAtomName(self, atomName):
        '''
        Remove numbers and quotes from atom names.
        '''
        return subn(self.__number, "", atomName.upper())[0]
                    
    def load(self, datasetFileName):
        ''' 
        Loads the dataset
    
        :type datasetFileName: string
        :param datasetFileName: the path to the dataset
        '''
        if (not os.path.isfile(datasetFileName)):
            raise Exception("%s is not a valid file name."%datasetFileName)

        fin  = open(datasetFileName,'r')

        line  = fin.readline()
        
        atomsNames      = []
        atoms           = []
        atomPositions   = []
        atomAminoAcides = []
        atomCharges     = []
        atomBonds       = []
        
        readingAtoms = False
        readingBonds = False
        
        moleculeName  = ""
        while line:
            line = line.strip()
            if line.startswith("#"):
                line = fin.readline()
                continue
            
            # if a new molecule/protein was found
            if line.startswith("@<TRIPOS>MOLECULE"):
                # stores the data of the molecule that just ended
                if len(atoms) > 0:
                    self._moleculeAtoms.append(numpy.asarray(atoms,numpy.int32))
                    self._moleculeAtomNames.append(numpy.asarray(atomsNames,numpy.int32))
                    self._moleculeAtomPositions.append(numpy.asarray(atomPositions,numpy.float32))
                    self._moleculeNames.append(moleculeName)
                    self._moleculeAtomAminoAcides.append(numpy.asarray(atomAminoAcides,numpy.int32))
                    self._moleculeAtomCharges.append(numpy.asarray(atomCharges,numpy.float32))
                    self._moleculeAtomBonds.append(numpy.asarray(atomBonds,numpy.int32))
                
                atomsNames      = []
                atoms           = []
                atomPositions   = []
                atomAminoAcides = []
                atomCharges     = []
                atomBonds       = []
                
                # reads the molecule name 
                moleculeName = fin.readline().strip()
                readingAtoms = False
                readingBonds = False
                                
            # if the ATOMS sections was found
            elif line.startswith("@<TRIPOS>ATOM"):
                readingAtoms = True
                readingBonds = False
            # if the BONDs sections was found
            elif line.startswith("@<TRIPOS>BOND"):
                readingAtoms = False
                readingBonds = True
            # if the SUBSTRUCTURE sections was found
            elif line.startswith("@<TRIPOS>SUBSTRUCTURE"):
                readingAtoms = False
                readingBonds = False
            elif line.startswith("@<TRIPOS>SOLVATION"):
                readingAtoms = False
                readingBonds = False
            # reads data from section: @<TRIPOS>ATOM
            elif readingAtoms and len(line) > 1:
                # 8 fields: [id, atom_name, posX, posY, posZ, other, aminoId, amino_acid, charge]
                data = line.split()
                # reads cleaned atom name (without numbers)
                atoms.append(self.getTermIndexAdd(self.cleanAtomName(data[1])))
                # reads atom name
                atomsNames.append(self.getTermIndexAdd(data[1]))
                # reads atom position
                atomPositions.append([float(data[2]), float(data[3]), float(data[4])])
                # reads the amino acid name
                atomAminoAcides.append(self.getTermIndexAdd(data[7]))
                # reads the charge
                atomCharges.append(float(data[8]))
            # reads data from section: @<TRIPOS>BOND
            elif readingBonds and len(line) > 1:
                # 4 fields: [id, atom1_id, atom2_id, ?]
                data = line.split()
                # we must subtract 1 because we store atoms from the position 0
                # while the position in the file start with 1
                atomBonds.append([int(data[1])-1, int(data[2])-1])
            
            line = fin.readline()
        fin.close()

        if len(atoms) > 0:
            self._moleculeAtoms.append(numpy.asarray(atoms,numpy.int32))
            self._moleculeAtomNames.append(numpy.asarray(atomsNames,numpy.int32))
            self._moleculeAtomPositions.append(numpy.asarray(atomPositions,numpy.float32))
            self._moleculeNames.append(moleculeName)
            self._moleculeAtomAminoAcides.append(numpy.asarray(atomAminoAcides,numpy.int32))
            self._moleculeAtomCharges.append(numpy.asarray(atomCharges,numpy.float32))
            self._moleculeAtomBonds.append(numpy.asarray(atomBonds,numpy.int32))
        
#         # creates a KDTree for each molecule using its atom positions
#         # these KDTrees speed up the search for neighbor atoms 
#         self.createMoleculeKDTrees()
        
    def save(self, datasetFileName):
        ''' 
        Saves the dataset
        '''
        pass

    def createMoleculeKDTrees(self):
        ''' 
        Creates a KDTree for each molecule using its atom positions. 
        '''
        self._moleculeAtomKDTrees = []
        for atomPositions in self._moleculeAtomPositions:
            self._moleculeAtomKDTrees.append(spatial.KDTree(atomPositions))

if __name__ == '__main__':
    from time import time
    
#     inputFileName  = "../../deepbio-data/dockoutput/tk/ZINC04225128_1.mol2"
#     inputFileName  = "../../deepbio-data/dockoutput/tk/virtual_flex_scored.mol2"
    inputFileName  = "../../deepbio-data/dockoutput/tk/rec.mol2"
    
    initTime = time()
    ds = PDBDataset()
    
    print "Loading dataset..."
    ds.load(inputFileName)
    print "# molecules", len(ds._moleculeNames)
    
    print "Creating KDTress"
    ds.createMoleculeKDTrees()
    print "# molecule atom kdtrees:", len(ds._moleculeAtomKDTrees)
