#!/bin/bash

rm -rf ./DM_V1_R_transfer_0/output
../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_0/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V1_R_transfer_0/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_transfer_0/configFiles/test/testConfig.cfg 

rm -rf ./DM_V1_R_transfer_1/output
../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_1/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V1_R_transfer_1/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_transfer_1/configFiles/test/testConfig.cfg 

rm -rf ./DM_V1_R_transfer_2/output
../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_2/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V1_R_transfer_2/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_transfer_2/configFiles/test/testConfig.cfg 

rm -rf ./DM_V1_R_transfer_3/output
../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_3/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V1_R_transfer_3/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_transfer_3/configFiles/test/testConfig.cfg 

rm -rf ./DM_V1_R_transfer_4/output
../deepMedicRun -dev cuda -train ./DM_V1_R_transfer_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_R_4/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V1_R_transfer_4/output/cnnModels/trainSession/*final*.save -test ./DM_V1_R_transfer_4/configFiles/test/testConfig.cfg
