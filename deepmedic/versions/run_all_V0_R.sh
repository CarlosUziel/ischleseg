#!/bin/bash

rm -rf ./DM_V0_R_0/output
../deepMedicRun -dev cuda -newModel ./DM_V0_R_0/configFiles/model/modelConfig.cfg -train ./DM_V0_R_0/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V0_R_0/output/cnnModels/trainSession/*final*.save -test ./DM_V0_R_0/configFiles/test/testConfig.cfg 

rm -rf ./DM_V0_R_1/output
../deepMedicRun -dev cuda -newModel ./DM_V0_R_1/configFiles/model/modelConfig.cfg -train ./DM_V0_R_1/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V0_R_1/output/cnnModels/trainSession/*final*.save -test ./DM_V0_R_1/configFiles/test/testConfig.cfg 

rm -rf ./DM_V0_R_2/output
../deepMedicRun -dev cuda -newModel ./DM_V0_R_2/configFiles/model/modelConfig.cfg -train ./DM_V0_R_2/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V0_R_2/output/cnnModels/trainSession/*final*.save -test ./DM_V0_R_2/configFiles/test/testConfig.cfg 

rm -rf ./DM_V0_R_3/output
../deepMedicRun -dev cuda -newModel ./DM_V0_R_3/configFiles/model/modelConfig.cfg -train ./DM_V0_R_3/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V0_R_3/output/cnnModels/trainSession/*final*.save -test ./DM_V0_R_3/configFiles/test/testConfig.cfg 

rm -rf ./DM_V0_R_4/output
../deepMedicRun -dev cuda -newModel ./DM_V0_R_4/configFiles/model/modelConfig.cfg -train ./DM_V0_R_4/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V0_R_4/output/cnnModels/trainSession/*final*.save -test ./DM_V0_R_4/configFiles/test/testConfig.cfg 
