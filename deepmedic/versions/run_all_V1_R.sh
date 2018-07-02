#!/bin/bash

rm -rf ./DM_V1_R_0/output
../deepMedicRun -dev cuda -newModel ./DM_V1_R_0/configFiles/model/modelConfig.cfg -train ./DM_V1_R_0/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_R_0/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_0/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_0/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V1_R_1/output
../deepMedicRun -dev cuda -newModel ./DM_V1_R_1/configFiles/model/modelConfig.cfg -train ./DM_V1_R_1/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_R_1/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_1/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_1/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V1_R_2/output
../deepMedicRun -dev cuda -newModel ./DM_V1_R_2/configFiles/model/modelConfig.cfg -train ./DM_V1_R_2/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_R_2/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_2/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_2/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V1_R_3/output
../deepMedicRun -dev cuda -newModel ./DM_V1_R_3/configFiles/model/modelConfig.cfg -train ./DM_V1_R_3/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_R_3/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_3/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_3/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V1_R_4/output
../deepMedicRun -dev cuda -newModel ./DM_V1_R_4/configFiles/model/modelConfig.cfg -train ./DM_V1_R_4/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_R_4/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_4/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_4/output/cnnModels/trainSession/*final*.save
