"""
Created on May 6, 2018
@author: jana

Module purpose description
"""
# Python standard library
import random
import sys

# 3rd party modules
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

# Our own
from data_processing import (
    load_data,
    prepareMolData,
    prepareMinibatches,
    context_dictionary,
    loadProteinRestrictions,
)
from model import DeepVS
from scorer_auc_enrichment_factor import Scorer

# Naming schemes
regular_objects = "snake_case_lowercase_underscores"


class CamelCase(object):
    def __init__(self, *args):
        pass


camel_case_instance = "snake_case_again"


def snake_case_functions():
    pass


CONSTANT_UPPERCASE_UNDERSCORE = "This is a constant"


class DeepVSExperiment:
    def __init__(
        self,
        embedding_size=200,
        cf=400,
        h=50,
        lr=0.00001,
        kc=6,
        kp=2,
        num_epochs=7,
        minibatchSize=20,
        l2_reg_rate=0.0001,
        use_Adam=True,
    ):

        self.embedding_size = embedding_size
        self.cf = cf
        self.h = h
        self.lr = lr
        self.kc = kc
        self.kp = kp
        self.num_epochs = num_epochs
        self.minibatchSize = minibatchSize
        self.l2_reg_rate = l2_reg_rate
        self.use_Adam = use_Adam

        self._aucSumByEpoch = [0.0] * 10
        self._EfMaxSumByEpoch = [0.0] * 10
        self._Ef2SumByEpoch = [0.0] * 10
        self._Ef20SumByEpoch = [0.0] * 10
        self._numProteinProcessed = 0

    def run(
        self,
        dataset_path,
        proteins_names_training,
        proteins_names_test,
        proteinRestrictions,
    ):
        torch.manual_seed(31)
        rng = random.Random(31)
        self.epoch = 0
        self._numProteinProcessed += 1.0

        testProRestrictions = proteinRestrictions.get(proteinsNames_test[0])

        if testProRestrictions is not None:
            i = len(proteinsNames_training) - 1
            while i > -1:
                if proteinsNames_training[i] in testProRestrictions:
                    del proteinsNames_training[i]
                i -= 1

        # Preparing training dataset
        print "Loading data ..."
        molName_training, molClass_training, molData_training = load_data(
            datasetPath, self.kc, self.kp, proteinsNames_training, rng, randomize=True
        )
        print "proteinsNames_training: ", proteinsNames_training
        print "Preparing data ..."
        context_to_ix_training = context_dictionary(molData_training)
        molData_ix_training = prepareMolData(molData_training, context_to_ix_training)
        molDataBatches_training = prepareMinibatches(
            molData_ix_training, molClass_training, self.minibatchSize
        )

        # Preparing test dataset
        molName_test, molClass_test, molData_test = load_data(
            datasetPath, self.kc, self.kp, proteinsNames_test, rng, randomize=False
        )
        print "proteinsNames_test: ", proteinsNames_test
        print "number of test molecules: ", len(molData_test)
        molData_ix_test = prepareMolData(molData_test, context_to_ix_training)
        molDataBatches_test = prepareMinibatches(
            molData_ix_test, molClass_test, self.minibatchSize
        )

        # Number of columns in the embedding matrix
        vocab_size = len(context_to_ix_training)

        # Instantiate Model  Class
        model = DeepVS(
            vocab_size, self.embedding_size, self.cf, self.h, self.kc, self.kp
        )
        #####################
        # Use GPU for model #
        #####################
        if torch.cuda.is_available():
            model.cuda()
            print "using GPU!"

        # Instantiate Loss Class
        loss_fuction = nn.NLLLoss()

        # Instantiate scorer
        scorer = Scorer()

        # Instantiate optimizer class: using Adam
        if self.use_Adam:
            optimizer = optim.Adam(
                model.parameters(), self.lr, weight_decay=self.l2_reg_rate
            )
            print "using Adam"
        else:
            optimizer = optim.SGD(
                model.parameters(), self.lr, weight_decay=self.l2_reg_rate
            )
            print "using SGD"

        print "lr = ", self.lr
        print "ls_reg_rate = ", self.l2_reg_rate

        # Train Model
        print "Training ..."
        for epoch in range(1, self.num_epochs + 1):
            total_loss = 0.0
            model.train()
            for cmplx, cls, msk in molDataBatches_training:
                # convert contexts and classes into torch variables
                if torch.cuda.is_available():
                    cls = autograd.Variable(torch.LongTensor(cls).cuda())
                    cmplx = autograd.Variable(torch.LongTensor(cmplx).cuda())
                    mskv = autograd.Variable(torch.FloatTensor(msk).cuda())
                else:
                    cls = autograd.Variable(torch.LongTensor(cls))
                    cmplx = autograd.Variable(torch.LongTensor(cmplx))
                    mskv = autograd.Variable(torch.FloatTensor(msk))

                model.zero_grad()

                # Run the forwad pass
                log_probs = model(cmplx, mskv)

                # Compute loss and update model
                loss = loss_fuction(log_probs, cls)
                loss.backward()
                optimizer.step()
                total_loss += loss.data[0]

            # shuffles the training set after each epoch
            rng.shuffle(molDataBatches_training)

            # sets model to eval (needed to use dropout in eval mode)
            model.eval()
            # Test model after each epoch
            correct = 0.0
            numberOfMolecules = 0.0
            total_loss_test = 0.0
            scores = []
            testMolId = 0
            for cmplx_test, cls_test, msk_test in molDataBatches_test:
                cls_test = torch.LongTensor(cls_test)
                # convert contexts and classes into torch variables
                if torch.cuda.is_available():
                    cls_test_v = autograd.Variable(cls_test.cuda())
                    cmplx_test = autograd.Variable(torch.LongTensor(cmplx_test).cuda())
                    mskv_test = autograd.Variable(torch.FloatTensor(msk_test).cuda())
                else:
                    cls_test_v = autograd.Variable(cls_test)
                    cmplx_test = autograd.Variable(torch.LongTensor(cmplx_test))
                    mskv_test = autograd.Variable(torch.FloatTensor(msk_test))

                # Run the forwad pass
                outputs = model(cmplx_test, mskv_test)
                loss_test = loss_fuction(outputs, cls_test_v)

                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                for cur_scr, cur_cls in zip(np.exp(outputs.data[:, 1]), cls_test):
                    scores.append([cur_scr, cur_cls, molName_test[testMolId]])
                    testMolId += 1

                numberOfMolecules += cls_test.size()[0]

                if torch.cuda.is_available():
                    correct += (predicted.cpu() == cls_test.cpu()).sum()
                else:
                    correct += (predicted == cls_test).sum()

                total_loss_test += loss_test.data[0]

            accuracy = 100 * correct / numberOfMolecules
            print "--------------------------------------------------------------------------------------------"
            print "epoch = %d;  total loss training = %.4f; total loss test = %.4f; accuracy = %f" % (
                epoch,
                total_loss / len(molDataBatches_training),
                total_loss_test / len(molDataBatches_test),
                accuracy,
            )
            print (
                f"epoch = {epoch};  total loss training = {total_loss/len(molDataBatches_training):.4f}; total loss test = {total_loss_test/len(molDataBatches_test):.4f}; accuracy = {accuracy:.f}"
            )
            (
                efAll,
                dataForROCCurve,
                efValues,
                aucValue,
            ) = scorer.computeEnrichmentFactor_and_AUC(scores, removeRepetitions=True)
            self._aucSumByEpoch[self.epoch] += aucValue
            self._EfMaxSumByEpoch[self.epoch] += efValues[2]
            self._Ef2SumByEpoch[self.epoch] += efValues[0]
            self._Ef20SumByEpoch[self.epoch] += efValues[1]

            self.epoch += 1

        print "Average AUC, EF2, EF20, EFMax by epoch for %d proteins:" % self._numProteinProcessed
        for k in xrange(self.num_epochs):
            print "Ep: %d, AUC: %.4f -" % (
                k + 1,
                self._aucSumByEpoch[k] / self._numProteinProcessed,
            ),
        print " "
        for k in xrange(self.num_epochs):
            print "Ep: %d, EF 2%%: %.4f -" % (
                k + 1,
                self._Ef2SumByEpoch[k] / self._numProteinProcessed,
            ),
        print " "
        for k in xrange(self.num_epochs):
            print "Ep: %d, EF 20%%: %.4f -" % (
                k + 1,
                self._Ef20SumByEpoch[k] / self._numProteinProcessed,
            ),
        print " "
        for k in xrange(self.num_epochs):
            print "Ep: %d, EF Max: %.4f -" % (
                k + 1,
                self._EfMaxSumByEpoch[k] / self._numProteinProcessed,
            ),
        print " "
        sys.stdout.flush()


if __name__ == "__main__":

    """
    Definition of Hyperparameters:
    
    embedding_size = embedding size of d^atm, d^amino, d^chg, d^dist
    cf = number of convolutional filters
    h =  number of hidden units
    lr = learning rate
    kc = number of neighboring atoms from compound
    kp = number of neighboring atoms from protein
    num_epoch = number of epochs
    """

    dvsExp = DeepVSExperiment(
        embedding_size=200,
        cf=400,
        h=50,
        lr=0.00001,
        kc=6,
        kp=2,
        num_epochs=7,
        minibatchSize=20,
        l2_reg_rate=0.0001,
        use_Adam=True,
    )

    proteinNames = [
        "ace",
        "ache",
        "ada",
        "alr2",
        "ampc",
        "ar",
        "cdk2",
        "comt",
        "cox1",
        "cox2",
        "dhfr",
        "egfr",
        "er_agonist",
        "er_antagonist",
        "fgfr1",
        "fxa",
        "gart",
        "gpb",
        "gr",
        "hivpr",
        "hivrt",
        "hmga",
        "hsp90",
        "inha",
        "mr",
        "na",
        "p38",
        "parp",
        "pde5",
        "pdgfrb",
        "pnp",
        "ppar",
        "pr",
        "rxr",
        "sahh",
        "src",
        "thrombin",
        "tk",
        "trypsin",
        "vegfr2",
    ]

    datasetPath = "dud_vinaout_deepvs/"
    proteinGroupsFileName = "protein.groups"
    proteinCrossEnrichmentFileName = "protein.cross_enrichment"

    proteinRestrictions = loadProteinRestrictions(
        proteinGroupsFileName, proteinCrossEnrichmentFileName
    )

    for pName in proteinNames:
        proteinNames_test = []
        proteinNames_training = ""
        if pName in proteinNames:
            proteinNames_test.append(pName)
            proteinNames_training = proteinNames[:]
            del proteinNames_training[proteinNames_training.index(pName)]

            print "======================================================================"
            print "Experimental results for protein:", proteinNames_test
            print "======================================================================"
            dvsExp.run(
                datasetPath,
                proteinNames_training,
                proteinNames_test,
                proteinRestrictions,
            )
