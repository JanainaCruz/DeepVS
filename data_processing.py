'''
Created on May 6, 2018
@author: jana
'''
import os
import random



def load_data (datasetPath, kc, kp, proteinNames, rng, randomize = False):
    '''
    function load_data
    
    1. Create a function that will read the file "data" and create the following list:
    MolName = [name, name2, ..., name3] #type string
    
    MolClass = [1,0,1,0,...,0,0] # type integer
    
    MolData = [
    [[context_of_atom1],[context_of_atom2],[context_of_atom3], [context_of_atom4]],
    [[context_of_atom1], [context_of_atom2], [context_of_atom3]],
    [[context_of_atom1],[context_of_atom2], [context_of_atom3], [context_of_atom4],[context_of_atom5]],
    ] # type string

    '''
    MolName = []
    MolClass = []
    MolData = []   
    for proteinName in  proteinNames:
        file_name = os.path.join(datasetPath, "%s.deepvs"%proteinName)
        f = open(file_name)
        for line in f:
            line = line.strip() 
            if len(line) != 0: 
                data = line.split(",")
                
                if data[0][0] == "@":
                    MolData.append([data[0], int(data[1])])
                else:
                    i_atm = 0
                    for ligand in data[0:kc+kp]:
                        data[i_atm] = ligand + "_atm"
                        i_atm += 1
                    i_dist = kc+kp
                    for ligand in data[kc+kp:(kc+kp)*2]:
                        data[i_dist] = str(ligand) + "_dist"
                        i_dist += 1    
                    i_chr = (kc+kp)*2
                    for ligand in data[(kc+kp)*2:(kc+kp)*3]:
                        data[i_chr] = str(ligand) + "_chr"
                        i_chr += 1   
                    i_amino = (kc+kp)*3
                    for ligand in data[(kc+kp)*3:(kc+kp)*3+kp]:
                        data[i_amino] = str(ligand) + "_amino"
                        i_amino += 1 
                                                               
                    MolData[-1].append(data)   
                               
        f.close()
        
    if randomize:
        rng.shuffle(MolData)
        
    for ligand in MolData:
        MolName.append(ligand[0])
        MolClass.append(ligand[1]) 
        del ligand[0], ligand[0]
    return MolName, MolClass, MolData


def context_dictionary(molData):
    ''' 
    Create a dictionary using the values in molecule contexts 
    '''
    context_to_ix = {'UNK':0}
    ix = 1
    
    for ligand in molData:
        for context in ligand:
            for position in context:
                if position not in context_to_ix:
                    context_to_ix[position] = ix
                    ix += 1    
    
    return context_to_ix


def prepareMolData(molData, context_to_ix):
    '''
    Replaces the information in the MolData to the respective key in the dictionary and create the input data to neural network
    '''
    unk_idx = context_to_ix['UNK']
    temp_MolData = []
    for ligand in molData:
        temp_ligand = []
        temp_MolData.append(temp_ligand)
        for context in ligand:
            temp_context = []
            temp_ligand.append(temp_context)
            for position in context:
                temp_context.append(context_to_ix.get(position, unk_idx))
                       
    return temp_MolData

    
def prepareMinibatches(molData_ix, molClass, minibatchSize):
    '''
    Creates minibatches
    '''
    new_molData   = []
    current_Data  = []
    current_Class = []
    
    for m, c in zip(molData_ix, molClass):
        current_Data.append(m)
        current_Class.append(c)
        if len(current_Data) == minibatchSize:
            new_molData.append([current_Data, current_Class])
            current_Data = []
            current_Class = []
    
    if len(current_Data) > 0:
        new_molData.append([current_Data, current_Class])
    
    # creates the fake molecule contexts:  
    fake = [0]  * len(new_molData[0][0][0][0]) # indexes meaning new_molData[data][batch][molecule][context]
    # make sure that molecules in each minibatch has the same size
    for minibatch in new_molData:
        minibatchData = minibatch[0] # gets the data only. minibatch[1] contains the classes
        mask_of_minibatch = []
        largest_size = 0
        for mol in minibatchData:
            if len(mol) > largest_size:
                largest_size = len(mol)
                               
        # adds fake molecules 
        for mol in minibatchData:
            mask_of_minibatch.append([0]* len(mol) + [-999]*(largest_size - len(mol)))
            mol.extend([fake]*(largest_size - len(mol)))

        minibatch.append(mask_of_minibatch)
    
    # each element in new_molData is a minibatch and contains: [data, classes, mask] 
    return new_molData


def loadProteinRestrictions(proteinGroupsFileName, proteinCrossEnrichmentFileName):
    """
    Loads the restrictions of each protein.
    """
    proteinRestrictions = {}
    if proteinGroupsFileName != None:
        f = open(proteinGroupsFileName, "r")
        for line in f:
            proteins = line.strip().split(",")
            
            for p in proteins:
                pgroup = {}               
                
                for p2 in proteins:
                    if p2 != p:
                        pgroup[p2] = True                         
                
                pgroupTemp = proteinRestrictions.get(p, {})
                #print p
                #print pgroupTemp
                for p2 in pgroup.iterkeys():
                    pgroupTemp[p2] = True
                    #print pgroupTemp
                proteinRestrictions[p] = pgroupTemp 
                #print proteinRestrictions
        f.close()
     
    if proteinCrossEnrichmentFileName != None:
        f = open(proteinCrossEnrichmentFileName, "r")
        for line in f:
            proteins = line.strip().split(",")
            pgroup = proteinRestrictions.get(proteins[0])
            #print p
            #print pgroup
            for p in proteins[1:]:
                pgroup[p] = True 
        f.close()
     
 
    return proteinRestrictions
        

if __name__ == '__main__':
    datasetPath  = '/home/jana/pytorch_classes/Pytorch_classes/deepVS_exercise/dataset'
    
    rng = random.Random(31)
    minibatchSize = 2
    kc = 6
    kp = 2 
    proteinNames_traning = [ 'ache', 'ampc', 'ada']
    proteinNames_test = ['ar']
    #proteinNames = [ 'ache', 'ada', 'alr2', 'ampc', 'ar']
    
    molName_training, molClass_training, molData_training = load_data(datasetPath, kc, kp, proteinNames_traning, rng, randomize = True)
    context_to_ix_training = context_dictionary(molData_training)
    molData_ix_training = prepareMolData(molData_training, context_to_ix_training)
    molDataBatches_training, molClassBatches_training, mask_training = prepareMinibatches(molData_ix_training, molClass_training, minibatchSize)
    #print molName_training
    print molClass_training
    print molData_training
    
    molName_test, molClass_test, molData_test = load_data(datasetPath, kc, kp, proteinNames_test, rng, randomize = False)
    molData_ix_test = prepareMolData(molData_test, context_to_ix_training)
    molDataBatches_test, molClassBatches_test, mask_test = prepareMinibatches(molData_ix_test, molClass_test, minibatchSize)
    print molClassBatches_test




