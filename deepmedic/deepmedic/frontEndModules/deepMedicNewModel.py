# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange
import os

from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import *
from deepmedic.frontEndModules.frontEndHelpers.createModelParametersClass import CreateModelSessionParameters
from deepmedic.frontEndModules.frontEndHelpers.preparationForSessionHelpers import makeFoldersNeededForCreateModelSession

from deepmedic.loggingAndMonitoring import logger
from deepmedic.neuralnet.cnn3d import Cnn3d
from deepmedic.neuralnet.utils import load_object_from_gzip_file, dump_cnn_to_gzip_file_dotSave
from deepmedic.loggingAndMonitoring.utils import datetimeNowAsStr


class ModelConfig(object):
    configStruct = {} #In here will be placed all read arguments.
    def get(self,string1) :
        return self.configStruct[string1] if string1 in self.configStruct else None
    
    #Optional but highly suggested.
    MODEL_NAME = "modelName"
    #[REQUIRED] Output:
    FOLDER_FOR_OUTPUT = "folderForOutput" #MUST BE GIVEN
    
    #================ MODEL PARAMETERS =================
    NUMB_CLASSES = "numberOfOutputClasses"
    NUMB_INPUT_CHANNELS_NORMAL = "numberOfInputChannels"
    
    #===Normal pathway===
    N_FMS_NORM = "numberFMsPerLayerNormal"
    KERN_DIM_NORM = "kernelDimPerLayerNormal"
    RESID_CONN_LAYERS_NORM = "layersWithResidualConnNormal"
    LOWER_RANK_LAYERS_NORM = "lowerRankLayersNormal"
    
    #==Subsampled pathway==
    USE_SUBSAMPLED = "useSubsampledPathway"
    #The below should be mirroring the pathway, otherwise let them specify them but throw warning all around that receptive field should stay the same!
    N_FMS_SUBS = "numberFMsPerLayerSubsampled"
    KERN_DIM_SUBS = "kernelDimPerLayerSubsampled"
    SUBS_FACTOR = "subsampleFactor"
    RESID_CONN_LAYERS_SUBS = "layersWithResidualConnSubsampled"
    LOWER_RANK_LAYERS_SUBS = "lowerRankLayersSubsampled"
    
    #==Extra hidden FC Layers. Final Classification layer is not included in here.
    N_FMS_FC = "numberFMsPerLayerFC"
    KERN_DIM_1ST_FC = "kernelDimFor1stFcLayer"
    RESID_CONN_LAYERS_FC = "layersWithResidualConnFC"
    
    #Size of Image Segments
    SEG_DIM_TRAIN = "segmentsDimTrain"
    SEG_DIM_VAL = "segmentsDimVal"
    SEG_DIM_INFERENCE = "segmentsDimInference"
    
    #==Batch Sizes===
    #Required.
    BATCH_SIZE_TR = "batchSizeTrain"
    BATCH_SIZE_VAL = "batchSizeVal"
    BATCH_SIZE_INFER = "batchSizeInfer"
    
    #Dropout Rates:
    DROP_R_NORM = "dropoutRatesNormal"
    DROP_R_SUBS = "dropoutRatesSubsampled"
    DROP_R_FC = "dropoutRatesFc"
    
    #Initialization method of the kernel weights.
    CONV_W_INIT = "convWeightsInit"
    #Activation Function for all convolutional layers:
    ACTIV_FUNCTION = "activationFunction"
    
    #Batch Normalization
    BN_ROLL_AV_BATCHES = "rollAverageForBNOverThatManyBatches"


def checkForDeprecatedConfig(configGet):
    msg_part1 = "ERROR: Deprecated input to the config: ["
    msg_part2 = "]. Please update config and use the new corresponding variable "
    msg_part3 = "]. Exiting."
    if configGet("initializeClassic0orDelving1") is not None:
        logger.print3(msg_part1 + "initializeClassic0orDelving1" + msg_part2 + "convWeightsInit" + msg_part3); exit(1)
    if configGet("relu0orPrelu1") is not None:
        logger.print3(msg_part1 + "relu0orPrelu1" + msg_part2 + "activationFunction" + msg_part3); exit(1)
    
    
#The argument should be absolute path to the config file for the model to create.
def deepMedicNewModelMain(modelConfigFilepath, absPathToPreTrainedModelGivenInCmdLine, listOfLayersToTransfer) :
    print("Given configuration file: ", modelConfigFilepath)
    #Parse the config file in this naive fashion...
    modelConfig = ModelConfig()
    exec(open(modelConfigFilepath).read(), modelConfig.configStruct)
    configGet = modelConfig.get #Main interface
    
    # Check for deprecated input
    checkForDeprecatedConfig(configGet)
    
    #Create Folders and Logger
    mainOutputAbsFolder = getAbsPathEvenIfRelativeIsGiven(configGet(modelConfig.FOLDER_FOR_OUTPUT), modelConfigFilepath)
    modelName = configGet(modelConfig.MODEL_NAME) if configGet(modelConfig.MODEL_NAME) else CreateModelSessionParameters.getDefaultModelName()
    [folderForCnnModels,
    folderForLogs] = makeFoldersNeededForCreateModelSession(mainOutputAbsFolder, modelName)
    loggerFileName = folderForLogs + "/" + modelName + ".txt"
    sessionLogger = logger.Logger(loggerFileName)
    
    sessionLogger.print3("CONFIG: Given THEANO_FLAGS: " + str(os.environ['THEANO_FLAGS']))
    sessionLogger.print3("CONFIG: The configuration file for the session was loaded from: " + str(modelConfigFilepath))
    
    #Fill in the session's parameters.
    createModelSessionParameters = CreateModelSessionParameters(
                    cnnModelName=modelName,
                    sessionLogger=sessionLogger,
                    mainOutputAbsFolder=mainOutputAbsFolder,
                    folderForSessionCnnModels=folderForCnnModels,
                    #===MODEL PARAMETERS===
                    numberClasses=configGet(modelConfig.NUMB_CLASSES),
                    numberOfInputChannelsNormal=configGet(modelConfig.NUMB_INPUT_CHANNELS_NORMAL),
                    #===Normal pathway===
                    numFMsNormal=configGet(modelConfig.N_FMS_NORM),
                    kernDimNormal=configGet(modelConfig.KERN_DIM_NORM),
                    residConnAtLayersNormal=configGet(ModelConfig.RESID_CONN_LAYERS_NORM),
                    lowerRankLayersNormal=configGet(ModelConfig.LOWER_RANK_LAYERS_NORM),
                    #==Subsampled pathway==
                    useSubsampledBool=configGet(modelConfig.USE_SUBSAMPLED),
                    numFMsSubsampled=configGet(modelConfig.N_FMS_SUBS),
                    kernDimSubsampled=configGet(modelConfig.KERN_DIM_SUBS),
                    subsampleFactor=configGet(modelConfig.SUBS_FACTOR),
                    residConnAtLayersSubsampled=configGet(ModelConfig.RESID_CONN_LAYERS_SUBS),
                    lowerRankLayersSubsampled=configGet(ModelConfig.LOWER_RANK_LAYERS_SUBS),
                    #==FC Layers====
                    numFMsFc=configGet(modelConfig.N_FMS_FC),
                    kernelDimensionsFirstFcLayer=configGet(modelConfig.KERN_DIM_1ST_FC),
                    residConnAtLayersFc=configGet(ModelConfig.RESID_CONN_LAYERS_FC),
                    #==Size of Image Segments ==
                    segmDimTrain=configGet(modelConfig.SEG_DIM_TRAIN),
                    segmDimVal=configGet(modelConfig.SEG_DIM_VAL),
                    segmDimInfer=configGet(modelConfig.SEG_DIM_INFERENCE),
                    #== Batch Sizes ==
                    batchSizeTrain=configGet(modelConfig.BATCH_SIZE_TR),
                    batchSizeVal=configGet(modelConfig.BATCH_SIZE_VAL),
                    batchSizeInfer=configGet(modelConfig.BATCH_SIZE_INFER),
                    #===Other Architectural Parameters ===
                    activationFunc=configGet(modelConfig.ACTIV_FUNCTION),
                    #==Dropout Rates==
                    dropNormal=configGet(modelConfig.DROP_R_NORM),
                    dropSubsampled=configGet(modelConfig.DROP_R_SUBS),
                    dropFc=configGet(modelConfig.DROP_R_FC),
                    #== Weight Initialization==
                    convWInitMethod=configGet(modelConfig.CONV_W_INIT),
                    #== Batch Normalization ==
                    bnRollingAverOverThatManyBatches=configGet(modelConfig.BN_ROLL_AV_BATCHES),
                    )
    
    
    createModelSessionParameters.sessionLogger.print3("\n===========    NEW CREATE-MODEL SESSION    ============")
    createModelSessionParameters.printParametersOfThisSession()
    
    createModelSessionParameters.sessionLogger.print3("\n=========== Creating the CNN model ===============")
    cnn3dInstance = Cnn3d()
    cnn3dInstance.make_cnn_model(*createModelSessionParameters.getTupleForCnnCreation())
    
    if absPathToPreTrainedModelGivenInCmdLine != None: # Transfer parameters from a previously trained model to the new one.
        createModelSessionParameters.sessionLogger.print3("\n=========== Pre-training the new model ===============")
        sessionLogger.print3("...Loading the pre-trained network. This can take a few minutes if the model is big...")
        cnnPretrainedInstance = load_object_from_gzip_file(absPathToPreTrainedModelGivenInCmdLine)
        sessionLogger.print3("The pre-trained model was loaded successfully from: " + str(absPathToPreTrainedModelGivenInCmdLine))
        from deepmedic.neuralnet.transferParameters import transferParametersBetweenModels
        cnn3dInstance = transferParametersBetweenModels(sessionLogger, cnn3dInstance, cnnPretrainedInstance, listOfLayersToTransfer)
    
    createModelSessionParameters.sessionLogger.print3("\n=========== Saving the model ===============")
    if absPathToPreTrainedModelGivenInCmdLine != None:
        filenameAndPathToSaveModel = createModelSessionParameters.getPathAndFilenameToSaveModel() + ".initial.pretrained." + datetimeNowAsStr()
    else:
        filenameAndPathToSaveModel = createModelSessionParameters.getPathAndFilenameToSaveModel() + ".initial." + datetimeNowAsStr()
    filenameAndPathWhereModelWasSaved =  dump_cnn_to_gzip_file_dotSave(cnn3dInstance, filenameAndPathToSaveModel, sessionLogger)
    createModelSessionParameters.sessionLogger.print3("=========== Creation of the model: \"" + str(createModelSessionParameters.cnnModelName) +"\" finished =================")
    
    return (cnn3dInstance, filenameAndPathWhereModelWasSaved)

