#!/bin/bash

rm -rf ./DM_V1_0/output
../deepMedicRun -dev cuda -newModel ./DM_V1_0/configFiles/model/modelConfig.cfg -train ./DM_V1_0/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_0/output/cnnModels/trainSession/*final*.save -test ./DM_V1_0/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V1_0/output/cnnModels/trainSession/*final*.save -test ./DM_V1_0/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V1_1/output
../deepMedicRun -dev cuda -newModel ./DM_V1_1/configFiles/model/modelConfig.cfg -train ./DM_V1_1/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_1/output/cnnModels/trainSession/*final*.save -test ./DM_V1_1/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V1_1/output/cnnModels/trainSession/*final*.save -test ./DM_V1_1/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V1_2/output
../deepMedicRun -dev cuda -newModel ./DM_V1_2/configFiles/model/modelConfig.cfg -train ./DM_V1_2/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_2/output/cnnModels/trainSession/*final*.save -test ./DM_V1_2/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V1_2/output/cnnModels/trainSession/*final*.save -test ./DM_V1_2/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V1_3/output
../deepMedicRun -dev cuda -newModel ./DM_V1_3/configFiles/model/modelConfig.cfg -train ./DM_V1_3/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_3/output/cnnModels/trainSession/*final*.save -test ./DM_V1_3/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V1_3/output/cnnModels/trainSession/*final*.save -test ./DM_V1_3/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V1_4/output
../deepMedicRun -dev cuda -newModel ./DM_V1_4/configFiles/model/modelConfig.cfg -train ./DM_V1_4/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_4/output/cnnModels/trainSession/*final*.save -test ./DM_V1_4/configFiles/test/testConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V1_4/output/cnnModels/trainSession/*final*.save -test ./DM_V1_4/configFiles/validation/valConfig.cfg 
