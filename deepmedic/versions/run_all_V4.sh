#!/bin/bash

#rm -rf ./DM_V4_0/output
../deepMedicRun -dev cuda -train ./DM_V4_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_0/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V4_0/output/cnnModels/trainSession/*final*.save -test ./DM_V4_0/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V4_0/output/cnnModels/trainSession/*final*.save -test ./DM_V4_0/configFiles/validation/valConfig.cfg 

#rm -rf ./DM_V4_1/output
../deepMedicRun -dev cuda -train ./DM_V4_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_1/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V4_1/output/cnnModels/trainSession/*final*.save -test ./DM_V4_1/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V4_1/output/cnnModels/trainSession/*final*.save -test ./DM_V4_1/configFiles/validation/valConfig.cfg 

#rm -rf ./DM_V4_2/output
../deepMedicRun -dev cuda -train ./DM_V4_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_2/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V4_2/output/cnnModels/trainSession/*final*.save -test ./DM_V4_2/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V4_2/output/cnnModels/trainSession/*final*.save -test ./DM_V4_2/configFiles/validation/valConfig.cfg 

#rm -rf ./DM_V4_3/output
../deepMedicRun -dev cuda -train ./DM_V4_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_3/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V4_3/output/cnnModels/trainSession/*final*.save -test ./DM_V4_3/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V4_3/output/cnnModels/trainSession/*final*.save -test ./DM_V4_3/configFiles/validation/valConfig.cfg 

#rm -rf ./DM_V4_4/output
../deepMedicRun -dev cuda -train ./DM_V4_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_4/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V4_4/output/cnnModels/trainSession/*final*.save -test ./DM_V4_4/configFiles/test/testConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V4_4/output/cnnModels/trainSession/*final*.save -test ./DM_V4_4/configFiles/validation/valConfig.cfg 
