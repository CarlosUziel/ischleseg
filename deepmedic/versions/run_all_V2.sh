#!/bin/bash

rm -rf ./DM_V2_0/output
../deepMedicRun -dev cuda -newModel ./DM_V2_0/configFiles/model/modelConfig.cfg -train ./DM_V2_0/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V2_0/output/cnnModels/trainSession/*final*.save -test ./DM_V2_0/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V2_transfer_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V2_0/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V2_1/output
../deepMedicRun -dev cuda -newModel ./DM_V2_1/configFiles/model/modelConfig.cfg -train ./DM_V2_1/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V2_1/output/cnnModels/trainSession/*final*.save -test ./DM_V2_1/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V2_transfer_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V2_1/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V2_2/output
../deepMedicRun -dev cuda -newModel ./DM_V2_2/configFiles/model/modelConfig.cfg -train ./DM_V2_2/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V2_2/output/cnnModels/trainSession/*final*.save -test ./DM_V2_2/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V2_transfer_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V2_2/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V2_3/output
../deepMedicRun -dev cuda -newModel ./DM_V2_3/configFiles/model/modelConfig.cfg -train ./DM_V2_3/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V2_3/output/cnnModels/trainSession/*final*.save -test ./DM_V2_3/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V2_transfer_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V2_3/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V2_4/output
../deepMedicRun -dev cuda -newModel ./DM_V2_4/configFiles/model/modelConfig.cfg -train ./DM_V2_4/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V2_4/output/cnnModels/trainSession/*final*.save -test ./DM_V2_4/configFiles/test/testConfig.cfg 
#../deepMedicRun -dev cuda -train ./DM_V2_transfer_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V2_4/output/cnnModels/trainSession/*final*.save