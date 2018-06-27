#!/bin/bash

../deepMedicRun -dev cuda -newModel ./DM_V1_0/configFiles/model/modelConfig.cfg -train ./DM_V1_0/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_0/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_1/configFiles/model/modelConfig.cfg -train ./DM_V1_1/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_1/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_2/configFiles/model/modelConfig.cfg -train ./DM_V1_2/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_2/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_3/configFiles/model/modelConfig.cfg -train ./DM_V1_3/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_3/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_4/configFiles/model/modelConfig.cfg -train ./DM_V1_4/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_4/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_5/configFiles/model/modelConfig.cfg -train ./DM_V1_5/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_5/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_5/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_6/configFiles/model/modelConfig.cfg -train ./DM_V1_6/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_6/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_6/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_7/configFiles/model/modelConfig.cfg -train ./DM_V1_7/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_7/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_7/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_8/configFiles/model/modelConfig.cfg -train ./DM_V1_8/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_8/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_8/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_9/configFiles/model/modelConfig.cfg -train ./DM_V1_9/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V1_transfer_9/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_9/output/cnnModels/trainSession/*final*.save
