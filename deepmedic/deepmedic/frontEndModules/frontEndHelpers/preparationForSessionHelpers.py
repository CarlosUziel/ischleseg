# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange
import os


def makeFoldersToSaveFilesForThisRun():
    #Create folders for saving the prediction images:
    parentFolderForSavingPredictedImages = folderToSavePredictionImages[:folderToSavePredictionImages.find("/")] #Usually should be ./predictions"
    if not os.path.exists(parentFolderForSavingPredictedImages) :
        os.mkdir(parentFolderForSavingPredictedImages)
    if not os.path.exists(folderToSavePredictionImages) : #The inner folder, for this particular run.
        os.mkdir(folderToSavePredictionImages)
    fullFolderPathToSavePredictedImagesFromDiceEvaluationOnValidationCasesDuringTraining = folderToSavePredictionImages + "/validationDuringTraining/"
    if not os.path.exists(fullFolderPathToSavePredictedImagesFromDiceEvaluationOnValidationCasesDuringTraining) :
        os.mkdir(fullFolderPathToSavePredictedImagesFromDiceEvaluationOnValidationCasesDuringTraining)
    fullFolderPathToSavePredictedImagesFromDiceEvaluationOnTestingCasesDuringTraining = folderToSavePredictionImages + "/testingDuringTraining/"
    if not os.path.exists(fullFolderPathToSavePredictedImagesFromDiceEvaluationOnTestingCasesDuringTraining) :
        os.mkdir(fullFolderPathToSavePredictedImagesFromDiceEvaluationOnTestingCasesDuringTraining)
    fullFolderPathToSavePredictedImagesDuringTesting = folderToSavePredictionImages + "/testing/"
    if not os.path.exists(fullFolderPathToSavePredictedImagesDuringTesting) :
        os.mkdir(fullFolderPathToSavePredictedImagesDuringTesting)
        
    folderWhereToPlaceVisualisationResults = folderToSavePredictionImages + "/visualisations/"
    if not os.path.exists(folderWhereToPlaceVisualisationResults) :
        os.mkdir(folderWhereToPlaceVisualisationResults)
        
    return [fullFolderPathToSavePredictedImagesFromDiceEvaluationOnValidationCasesDuringTraining,
            fullFolderPathToSavePredictedImagesFromDiceEvaluationOnTestingCasesDuringTraining,
            fullFolderPathToSavePredictedImagesDuringTesting,
            folderWhereToPlaceVisualisationResults]
    
def createMainOutputFolder(absMainOutputFolder) :
    if not os.path.exists(absMainOutputFolder) :
        os.mkdir(absMainOutputFolder)
        print("\t>>Created main output folder: ", absMainOutputFolder)
def createLogsFolder(folderForLogs) :
    if not os.path.exists(folderForLogs) :
        os.mkdir(folderForLogs)
        print("\t>>Created folder for logs: ", folderForLogs)
def createFolderForPredictions(folderForPredictions) :
    if not os.path.exists(folderForPredictions) :
        os.mkdir(folderForPredictions)
        print("\t>>Created folder for predictions: ", folderForPredictions)
def createFolderForSessionResults(folderForSessionResults) :
    if not os.path.exists(folderForSessionResults) :
        os.mkdir(folderForSessionResults)
        print("\t>>Created folder for session: ", folderForSessionResults)
def createFolderForSegmAndProbMaps(folderForSegmAndProbMaps) :
    if not os.path.exists(folderForSegmAndProbMaps) :
        os.mkdir(folderForSegmAndProbMaps)
        print("\t>>Created folder for segmentations and probability maps: ", folderForSegmAndProbMaps)
def createFolderForFeatures(folderForFeatures) :
    if not os.path.exists(folderForFeatures) :
        os.mkdir(folderForFeatures)
        print("\t>>Created folder for features: ", folderForFeatures)

def makeFoldersNeededForTestingSession(absMainOutputFolder, sessionName):
    #Create folders for saving the prediction images:
    print("Creating necessary folders for testing session...")
    createMainOutputFolder(absMainOutputFolder)
    
    folderForLogs = absMainOutputFolder + "/logs/"
    createLogsFolder(folderForLogs)
    
    folderForPredictions = absMainOutputFolder + "/predictions"
    createFolderForPredictions(folderForPredictions)
    
    folderForSessionResults = folderForPredictions + "/" + sessionName
    createFolderForSessionResults(folderForSessionResults)
    
    folderForSegmAndProbMaps = folderForSessionResults + "/predictions/"
    createFolderForSegmAndProbMaps(folderForSegmAndProbMaps)
    
    folderForFeatures = folderForSessionResults + "/features/"
    createFolderForFeatures(folderForFeatures)
    
    return [folderForLogs,
            folderForSegmAndProbMaps,
            folderForFeatures]

def createFolderForCnnModels(folderForCnnModels) :
    if not os.path.exists(folderForCnnModels) :
        os.mkdir(folderForCnnModels)
        print("\t>>Created folder to save cnn-models as they get trained: ", folderForCnnModels)

def createFolderForSessionCnnModels(folderForSessionCnnModels) :
    if not os.path.exists(folderForSessionCnnModels) :
        os.mkdir(folderForSessionCnnModels)
        print("\t>>Created folder to save session's cnn-models as they get trained: ", folderForSessionCnnModels)

def makeFoldersNeededForTrainingSession(absMainOutputFolder, sessionName):
    #Create folders for saving the prediction images:
    print("Creating necessary folders for training session...")
    createMainOutputFolder(absMainOutputFolder)
    
    folderForCnnModels = absMainOutputFolder + "/cnnModels/"
    createFolderForCnnModels(folderForCnnModels)
    
    folderForSessionCnnModels = folderForCnnModels + "/" + sessionName + "/"
    createFolderForSessionCnnModels(folderForSessionCnnModels)
    
    folderForLogs = absMainOutputFolder + "/logs/"
    createLogsFolder(folderForLogs)
    
    folderForPredictions = absMainOutputFolder + "/predictions"
    createFolderForPredictions(folderForPredictions)
    
    folderForSessionResults = folderForPredictions + "/" + sessionName
    createFolderForSessionResults(folderForSessionResults)
    
    folderForSegmAndProbMaps = folderForSessionResults + "/predictions/"
    createFolderForSegmAndProbMaps(folderForSegmAndProbMaps)
    
    folderForFeatures = folderForSessionResults + "/features/"
    createFolderForFeatures(folderForFeatures)
    
    return [folderForSessionCnnModels,
            folderForLogs,
            folderForSegmAndProbMaps,
            folderForFeatures]
    

def makeFoldersNeededForCreateModelSession(absMainOutputFolder, modelName):
    #Create folders for saving the prediction images:
    print("Creating necessary folders for create-new-model session...")
    createMainOutputFolder(absMainOutputFolder)
    
    folderForCnnModels = absMainOutputFolder + "/cnnModels/"
    createFolderForCnnModels(folderForCnnModels)
    
    folderForLogs = absMainOutputFolder + "/logs/"
    createLogsFolder(folderForLogs)
    
    return [folderForCnnModels,
            folderForLogs]


def checkCpuOrGpu(logger, compiledTheanoFunc):
    # compiledTheanoFunc: the return of theano.function( ), eg when compiling training or test functions.
    # Returns 1 if gpu, 0 if cpu.
    # From: http://www.deeplearning.net/software/theano_versions/dev/tutorial/using_gpu.html
    
    usingGpu = False
    usingCuDnn = False
    
    if any([x.op.__class__.__name__.startswith("Gpu") for x in compiledTheanoFunc.maker.fgraph.toposort()]):
        logger.print3('CONFIG: Theano is using the [GPU].')
        usingGpu = True    
        from theano.gpuarray.dnn import dnn_present
        usingCuDnn = dnn_present()
        logger.print3("CONFIG: Theano found and will use cuDNN ["+ str(usingCuDnn) +"]")
        
    else:
        logger.print3('CONFIG: Theano is using the [CPU].')
        usingGpu=False
        
    THEANO_FLAGS = "THEANO_FLAGS"
    if os.environ[THEANO_FLAGS].find("force_device=True") != -1 and os.environ[THEANO_FLAGS].find("device=cuda") != -1 and not usingGpu :
        logger.print3("ERROR:\t The THEANO_FLAGS, as set either in the environment or via the deepMedicRun script, they enforced the use of GPU.\n" +\
                      "\t However an internal check showed that Theano has not managed to start on the GPU.\n" +\
                      "\t Please check for any error messages thrown above by Theano (eg when loading or compiling the model) for indications about the problem.\n" +\
                      "\t Given THEANO_FLAGS= "+ str(os.environ[THEANO_FLAGS]) + "\n" +\
                      "\t Exiting!"); exit(1)
        
    
    
