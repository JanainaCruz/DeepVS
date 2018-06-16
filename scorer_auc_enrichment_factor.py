import numpy as np
import matplotlib.pyplot as plt

from auc_scorer import AUCScorer

class Scorer:

    def __init__(self):
        """
        Constructor.
        """
        self._aucSumByEpoch       = 0
        self._EfMaxSumByEpoch     = 0
        self._Ef2SumByEpoch       = 0
        self._Ef20SumByEpoch      = 0
        self._numProteinProcessed = 0

    def loadScoresFile(self, scoresFileName):
        """
        Loads scores.
        """
        scores = []
        f = open(scoresFileName, "r")
        for line in f:
            # line format: molecule_ID \t score \t active?
            data = line.strip().split("\t")
            if len(data) > 2:
                # scores formar: (score, isLigarnd?, moleculeName)
                scores.append([float(data[1]), int(data[2]), data[0]])
            else:
                print "Malformed line:",  data
        return scores

    def plotEnrichmentFactor(self, enrichmentFactorsAndLables, proteinLabel):
        """
        @param list enrichmentFactorsAndLables: a list of enrichment factor lists and their respective labels.
        """
        if len(enrichmentFactorsAndLables) < 1:
            return
        
        linestyles = [':', '--', '-', '-.']

        fig, ax = plt.subplots()
        i = 0
        for efList, efLabel in enrichmentFactorsAndLables:
            x  = []
            y  = []
            for perc, enrc in efList:
                x.append(perc)
                y.append(enrc)
                
            # x = np.linspace(0, 10, 500)
            # dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off
            
            line1, = ax.plot(x, y, linestyles[i], linewidth=2,
                             label=efLabel)
            i += 1
            #line1.set_dashes(dashes)
#            
        plt.xlabel('Percentual Selecionado', fontsize=18)
        plt.ylabel('Fator de Enriquecimento', fontsize=18)
        plt.title('Curvas de Enriquecimento - '+proteinLabel, fontsize=20, fontweight='bold')
        plt.legend(loc="lower right")
        plt.show()
        
        
    def plotROCCurve(self, dataPointsForROCCurve, proteinLabel):
        """
        @param list dataPointsForROCCurve: a list of data point lists and their respective labels.
        """
        if len(dataPointsForROCCurve) < 1:
            return
        
        linestyles = [':', '--', '-', '-.']
        
        fig, ax = plt.subplots()
        i = 0
        for fpAndTpPoints, efLabel in dataPointsForROCCurve:
            x  = fpAndTpPoints[0]  # false positive points
            y  = fpAndTpPoints[1]  # true positive points
                
            # x = np.linspace(0, 10, 500)
            # dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off
            
            line1, = ax.plot(x, y, linestyles[i], linewidth=2,
                             label=efLabel)

            i += 1
            
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Ranking Aleatorio')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos', fontsize=18)
        plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=18)
#         plt.title('Receiver Operating Characteristic (ROC) Curve - ' + proteinLabel)
        plt.title('Curvas ROC - ' + proteinLabel, fontsize=20, fontweight='bold')
        plt.legend(loc="lower right")
        plt.show()
        
        
    def computeEnrichmentFactor_and_AUC(self, scores, removeRepetitions = False):
        """
        Loads the active ligands.
        """
        scores.sort(reverse=True)
        
        if removeRepetitions:
            visitedMols = {}
            i = 0
            while i < len(scores):
                if scores[i][2] in visitedMols:
                    del scores[i]
                else:
                    visitedMols[scores[i][2]] = True
                    i += 1
        
        numLigants = 0
        for score_isActive_molName  in scores:
            if score_isActive_molName[1] == 1:
                numLigants += 1
        
        count = 0
        ef_denominator = float(numLigants) / len(scores)

        len_scores = float(len(scores))
        ef2   = int(round(len_scores * .02))
        ef20  = int(round(len_scores * .2))

        scoresToAuc = []        
        
        ef2Value    = 0
        ef20Value   = 0
        efMax       = 0
        efMax_count = 0
        efMax_ligandsFound = 0
        ligandsFound = 0
        
        allEnrichmentFactors = []
        efValues = []
        
        for score_isActive_molName in scores:
            count += 1
            if score_isActive_molName[1] == 1:
                ligandsFound += 1
            
            curEf = (ligandsFound / float(count)) / ef_denominator
            if efMax < curEf:
                efMax = curEf
                efMax_count = count
                efMax_ligandsFound = ligandsFound
            
            allEnrichmentFactors.append([(count / len_scores)*100, curEf])
            
            if count == ef2 or count == ef20:
                print "Ef@%d%%:"%(int(round((count/len_scores)*100))), curEf, ", #molecules:", count, ", #ligands found:", ligandsFound
            
            if count == ef2:
                ef2Value = curEf
                self._Ef2SumByEpoch  += curEf
                efValues.append(curEf)
            elif count == ef20:
                ef20Value = curEf
                self._Ef20SumByEpoch  += curEf
                efValues.append(curEf)

            if score_isActive_molName[1] == 1:
#                 scoresToAuc.append((1, abs(score_isActive_molName[0])/abs(scores[0][0])))
                scoresToAuc.append((1, score_isActive_molName[0]))
            else:
#                 scoresToAuc.append((0, abs(score_isActive_molName[0])/abs(scores[0][0])))
                scoresToAuc.append((0, score_isActive_molName[0]))

        self._EfMaxSumByEpoch += efMax
        efValues.append(efMax)

        print "EfMax:", efMax, ", #molecules:", efMax_count, ", #ligands found:", efMax_ligandsFound

        aucScorer = AUCScorer(scoresToAuc)
        print "AUC:", aucScorer.auc
        print "# ligands:", numLigants
        print "# molecules:", len(scores)
#         print "For paper(efMAx & ef2%% & ef20%% & auc): %.1f & %.1f & %.1f & %.2f"%(round(efMax,1), round(ef2Value,1), round(ef20Value,1), round(aucScorer.auc,2))
        
        self._aucSumByEpoch       += aucScorer.auc
        self._numProteinProcessed += 1
        
        dataForROCCurve = [aucScorer.falsePositiveRatePoints, aucScorer.truePositiveRatePoints] 

        return allEnrichmentFactors, dataForROCCurve, efValues, aucScorer.auc

    def printSummaryStatistics(self):
        """
        Prints the average AUC and EFs.
        """
        print ""
        print ""
        print "======================================================================"
        print "Average AUC, EF2, EF20, EFMax for %d proteins:"%self._numProteinProcessed
        print "======================================================================"
        print "EF Max: %.1f"%(round(self._EfMaxSumByEpoch/self._numProteinProcessed,1))
        print "EF 2%%: %.1f"%(round(self._Ef2SumByEpoch/self._numProteinProcessed,1))
        print "EF 20%%: %.1f"%(round(self._Ef20SumByEpoch/self._numProteinProcessed,1))
        print "AUC: %.2f"%(round(self._aucSumByEpoch/self._numProteinProcessed,2))
        

if __name__ == '__main__':
    from sys import argv
    import os
    inputFilesPath = argv[1]
    
    print "ResultsPath:", inputFilesPath 
    
#     ltest_proteinNames = ['ada', 'alr2', 'ar', 'comt', 'dhfr', 'er_antagonist', 
#                           'fxa', 'gart', 'hsp90', 'inha', 'p38', 'pnp', 'thrombin', 
#                           'tk']
    # DUD
#     ltest_proteinNames = ['ace', 'ache', 'ada', 'alr2', 'ampc', 'ar', 'cdk2', 'comt', 'cox1',
#                                'cox2', 'dhfr', 'egfr', 'er_agonist', 'er_antagonist', 
#                             'fgfr1', 'fxa', 'gart', 'gpb', 'gr', 'hivpr', 'hivrt', 'hmga', 'hsp90', 'inha', 'mr', 
#                             'na', 'p38', 'parp', 'pde5', 'pdgfrb', 'pnp', 'ppar', 'pr', 'rxr', 'sahh', 
#                             'src', 'thrombin', 'tk', 'trypsin', 'vegfr2']

    # DUD-E
    ltest_proteinNames = ['aa2ar', 'adrb1', 'adrb2', 'akt2', 'aofb', 'bace1', 'casp3', 'cp2c9', 'cp3a4', 'cxcr4', 'def', 
                              'drd3', 'esr2', 'fa7', 'fabp4', 'fak1', 'fkb1a', 'fnta', 'glcm', 'gria2', 'grik1', 'hxk4', 'ital',
                               'jak2', 'kif11', 'lkha4', 'mapk2', 'mk01', 'mk10', 'mmp13', 'mp2k1','nos1', 'pa2ga', 'plk1', 'ppara',
                               'ppard', 'ptn1', 'pyrd', 'reni', 'rock1', 'thb', 'tryb1', 'wee1', 'xiap']

    # cruzaina

    proteinLabel = ''

#     ltest_proteinNames = ['cys_desprotonada']
#     proteinLabel = '(His162)p/(Cys25)d'

    ltest_proteinNames = ['cys_neutra']
    proteinLabel = '(His162)p/(Cys25)n'

    scoreFileSuffixes  = [["vina", "Vina"], 
                          ["dock", "Dock6.6"], 
                          ["deepvs_vina", "DeepVS-ADV"], 
                          ["deepvs_dock", "DeepVS-Dock"]]

    scorer = Scorer()
    for pName in ltest_proteinNames:
        enrichmentFactors       = []
        dataPointsForROCCurve   = []
        for suffix in scoreFileSuffixes:
            proteinPath = os.path.join(inputFilesPath, "%s.%s.scores"%(pName, suffix[0]))
            if os.path.exists(proteinPath):
                print "======================================================================"
                print "Results for protein:", pName, proteinPath
                print "======================================================================"
                scores = scorer.loadScoresFile(proteinPath)
                efAll, dataForROCCurve, efValues, aucValue = scorer.computeEnrichmentFactor_and_AUC(scores)
                print suffix
                for line in efAll:
                    print line
                enrichmentFactors.append([efAll, suffix[1]])
                dataPointsForROCCurve.append([dataForROCCurve, suffix[1]])
#         scorer.plotEnrichmentFactor(enrichmentFactors, proteinLabel)
#         scorer.plotROCCurve(dataPointsForROCCurve, proteinLabel)
        
    print scorer.printSummaryStatistics()
    