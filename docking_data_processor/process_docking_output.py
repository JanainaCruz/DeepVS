from pdb_dataset import PDBDataset
from pdbqt_dataset import PDBQTDataset
from molecule_contexts_dataset import MoleculeContextsDataset
from util import traverseDirectory

import os

class PreprocessDockingOutput:

    def __init__(self, 
                 dockingProgram = 'vina'
                 ):
        """
        Constructor.
        """
        self.dockingProgram = dockingProgram
        
    def loadActiveLigands(self, fileName):
        """
        Loads active ligands.
        """
        ligands = {}
        f = open(fileName, "r")
        for line in f:
            l = line.strip()
            if len(l) > 1:
                lName = l
                k = 2
                while ligands.get(lName) != None:
                    lName = "%s%d"%(l,k)
                    k += 1
                ligands[lName] = True
        f.close()
        return ligands

    def processAllComplexesOfProtein(self, dataPath, proteinName, outputFileName,
                                     numIntraNeighbors, numInterNeighbors, 
                                     distIntervSize=.3, distMax=5, 
                                     chargeIntervSize=.05, chargeMin=-1, chargeMax=1):
        """
        Loads the contexts of all ligands and decoys of a given protein.
        """
        print "Loading ligands of", proteinName
        activeLigandsFileName = os.path.join(dataPath, "%s/ligands.list"%proteinName)
        activeLigands = self.loadActiveLigands(activeLigandsFileName)

        print "Loading protein and molecules of", proteinName
        if self.dockingProgram == 'dock':
            proteinFileName       = os.path.join(dataPath, "%s/rec.mol2"%proteinName)
            moleculesFileName     = os.path.join(dataPath, "%s/virtual_flex.mol2"%proteinName)
            dsMolecules = PDBDataset()
            
        elif self.dockingProgram == 'vina':
            proteinFileName       = os.path.join(dataPath, "%s/rec.pdbqt"%proteinName)
            moleculesFileName     = os.path.join(dataPath, "%s/vina_out"%proteinName)
            dsMolecules = PDBQTDataset()
            
        dsMolecules.load(proteinFileName)   # the protein must be the first item in the loaded dataset
        dsMolecules.load(moleculesFileName)
        
        print "Creating KDTrees..."
        dsMolecules.createMoleculeKDTrees()

        print "Creating contexts of protein:", proteinName
        molContextsDs = MoleculeContextsDataset(
                                        distanceDiscretizationIntervalSize = distIntervSize,
                                        largestDistance = distMax,
                                        chargeDiscretizationIntervalSize = chargeIntervSize,
                                        largestCharge  = chargeMax,
                                        smallestCharge = chargeMin)
        
        molContextsDs.createExamples(molecules = dsMolecules, 
                                     proteinId = 0, # the protein is the first molecule in the loaded dataset
                                     moleculeIdStart= 1, 
                                     moleculeIdEnd  = dsMolecules.getNumberOfMolecules(),  
                                     numIntraNeighbors = numIntraNeighbors, 
                                     numInterNeighbors = numInterNeighbors,
                                     ligands = activeLigands)
        
        molContextsDs.discretizeDistance()
        molContextsDs.discretizeCharges()
        
        molContextsDs.saveToFile(outputFileName)


    def loadVinaScores(self, proteinPath):
        '''
        Loads Vina scores. 
        '''
        self.vinaScoreAndMoleculeName = []
        traverseDirectory(os.path.join(proteinPath, "vina_out"), callback=self.extractScoreFromPDBQTFile, 
                              extension=".pdbqt")
            

    def extractScoreFromPDBQTFile(self, fileName):
        """
        Loads the scores from VINA output.
        """
        f = open(fileName, "r")
        #removes _1.pdbqt
        moleculeName = os.path.basename(fileName)[:-8]
        score = 0
        for line in f:
            if line.startswith("REMARK VINA RESULT:"):
                score = float(line[len("REMARK VINA RESULT:"):].strip().split()[0])
                break
        f.close()
        self.vinaScoreAndMoleculeName.append((score, moleculeName, len(self.vinaScoreAndMoleculeName)))
                
if __name__ == '__main__':
    dataPath  = '/media/jana/DATA/workspace/deepbio-data/vinaoutput'
    
    numIntraNeighbors    = 6
    numInterNeighbors    = 2
    distanceIntervalSize = .3
    distanceMax          = 5 
    chargeIntervalSize   = .05
    chargeMin            = -1
    chargeMax            = 1
    
    proteinsToProcess = ['ace', 'ache', 'ada', 'alr2', 'ampc', 'ar', 'cdk2','comt', 'cox1', 
                            'cox2', 'dhfr', 'egfr', 'er_agonist', 'er_antagonist', 
                            'fgfr1', 'fxa', 'gart', 'gpb', 'gr', 'hivpr', 'hivrt', 'hmga', 'hsp90', 'inha', 'mr', 
                            'na', 'p38', 'parp', 'pde5', 'pdgfrb', 'pnp', 'ppar', 'pr', 'rxr', 'sahh', 
                            'src', 'thrombin', 'tk', 'trypsin', 'vegfr2']
    
    preprocessor = PreprocessDockingOutput('vina')
    for proteinName in proteinsToProcess:
        outputFileName = os.path.join(dataPath, "%s.deepvs"%proteinName)
        preprocessor.processAllComplexesOfProtein(dataPath, 
                                         proteinName, 
                                         outputFileName,
                                         numIntraNeighbors, numInterNeighbors, 
                                         distIntervSize=distanceIntervalSize, 
                                         distMax=distanceMax, 
                                         chargeIntervSize=chargeIntervalSize, 
                                         chargeMin=chargeMin, 
                                         chargeMax=chargeMax)
