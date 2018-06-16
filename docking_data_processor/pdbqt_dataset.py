#! -*- coding: UTF-8 -*-
from pdb_dataset import PDBDataset
from util import traverseDirectory

import os
import numpy
from codecs import open


class PDBQTDataset(PDBDataset):
    """
    This class implements a dataset that can handle PDBQT files.
    """

    def load(self, datasetFileName):
        ''' 
        Loads the dataset
    
        :type datasetFileName: string
        :param datasetFileName: the path to the dataset
        '''
        if (not os.path.exists(datasetFileName)):
            raise Exception("%s is not a valid directory name."%datasetFileName)
        
        traverseDirectory(datasetFileName, self.loadPDBQTFile, extension=".pdbqt")

    def loadPDBQTFile(self, datasetFileName):
        ''' 
        Loads one pdbqt file
    
        :type datasetFileName: string
        :param datasetFileName: the path to the dataset
        '''

        fin  = open(datasetFileName,'r')

        line  = fin.readline()
        
        atomsNames      = []
        atoms           = []
        atomPositions   = []
        atomAminoAcides = []
        atomCharges     = []
        #atomBonds       = []
        moleculeName = os.path.basename(datasetFileName)[:-8]
        
        while line:
            line = line.strip()
            if line == "ENDMDL":
                break
            # reads data of one atom
            if line.startswith("ATOM "):
                # 12 fields: [keyword, id, atom_name, amino, ?, x, y, z, ?, ?, charge, ?]
                data = line.split()
                # reads cleaned atom name (without numbers)
                atoms.append(self.getTermIndexAdd(self.cleanAtomName(data[2])))
                # reads atom name
                atomsNames.append(self.getTermIndexAdd(data[2]))
                # reads atom position
                atomPositions.append([float(data[5]), float(data[6]), float(data[7])])
                # reads the amino acid name
                atomAminoAcides.append(self.getTermIndexAdd(data[3]))
                try:
                    # reads the charge
                    atomCharges.append(float(data[10]))
                except ValueError:
                    # sometimes the field 8 and 9 are collapsed, 
                    # in these cases we only have 11 fields
                    print "Error in:", datasetFileName
                    print "Line:", line
                    if len(data) == 11:
                        atomCharges.append(float(data[9]))
                    else:
                        atomCharges.append(0.0)
            
            line = fin.readline()
        fin.close()

        if len(atoms) > 0:
            self._moleculeAtoms.append(numpy.asarray(atoms,numpy.int32))
            self._moleculeAtomNames.append(numpy.asarray(atomsNames,numpy.int32))
            self._moleculeAtomPositions.append(numpy.asarray(atomPositions,numpy.float32))
            self._moleculeNames.append(moleculeName)
            self._moleculeAtomAminoAcides.append(numpy.asarray(atomAminoAcides,numpy.int32))
            self._moleculeAtomCharges.append(numpy.asarray(atomCharges,numpy.float32))
#             self._moleculeAtomBonds.append(numpy.asarray(atomBonds,numpy.int32))
        
if __name__ == '__main__':
    from time import time
    
    inputFileNameRec  = "../../deepbio-data/vinaoutput/comt/rec.pdbqt"
    inputFileNameMols = "../../deepbio-data/vinaoutput/comt/vina_out"
    
    initTime = time()
    ds = PDBQTDataset()
    
    print "Loading protein..."
    ds.load(inputFileNameRec)
    print "# molecules", len(ds._moleculeNames)

    print "Creating KDTress"
    ds.createMoleculeKDTrees()
    print "# molecule atom kdtrees:", len(ds._moleculeAtomKDTrees)
    
    i = 0
    for molName in ds._moleculeNames:
        print molName, "# atoms:", len(ds._moleculeAtoms[i])
        print "Atoms:"
        for atom in ds._moleculeAtoms[i]:
            print ds.getTermByIndex(atom),
        print
        print "charges:"
        for v in ds._moleculeAtomCharges[i]:
            print "%.4f"%v,
        print
        print "amino:"
        for v in ds._moleculeAtomAminoAcides[i]:
            print ds.getTermByIndex(v),
        print
        print "amino:"
        for v in ds._moleculeAtomPositions[i]:
            print v,
        print

