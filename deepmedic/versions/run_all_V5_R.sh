#!/bin/bash

rm -rf ./DM_V5_R_0/output
../deepMedicRun -dev cuda -train ./DM_V5_R_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_R_0/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V5_R_0/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_0/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V5_R_0/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_0/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V5_R_1/output
../deepMedicRun -dev cuda -train ./DM_V5_R_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_R_1/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V5_R_1/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_1/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V5_R_1/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_1/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V5_R_2/output
../deepMedicRun -dev cuda -train ./DM_V5_R_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_R_2/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V5_R_2/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_2/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V5_R_2/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_2/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V5_R_3/output
../deepMedicRun -dev cuda -train ./DM_V5_R_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_R_3/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V5_R_3/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_3/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V5_R_3/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_3/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V5_R_4/output
../deepMedicRun -dev cuda -train ./DM_V5_R_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_R_4/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V5_R_4/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_4/configFiles/test/testConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V5_R_4/output/cnnModels/trainSession/*final*.save -test ./DM_V5_R_4/configFiles/validation/valConfig.cfg 
