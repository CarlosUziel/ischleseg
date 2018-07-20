#!/bin/bash

rm -rf ./DM_V0_transfer_0/output
../deepMedicRun -dev cuda -train ./DM_V0_transfer_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_0/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V0_transfer_0/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_0/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V0_transfer_0/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_0/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V0_transfer_1/output
../deepMedicRun -dev cuda -train ./DM_V0_transfer_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_1/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V0_transfer_1/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_1/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V0_transfer_1/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_1/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V0_transfer_2/output
../deepMedicRun -dev cuda -train ./DM_V0_transfer_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_2/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V0_transfer_2/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_2/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V0_transfer_2/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_2/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V0_transfer_3/output
../deepMedicRun -dev cuda -train ./DM_V0_transfer_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_3/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V0_transfer_3/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_3/configFiles/test/testConfig.cfg 
../deepMedicRun -dev cuda -model ./DM_V0_transfer_3/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_3/configFiles/validation/valConfig.cfg 

rm -rf ./DM_V0_transfer_4/output
../deepMedicRun -dev cuda -train ./DM_V0_transfer_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_4/output/cnnModels/trainSession/*final*.save
../deepMedicRun -dev cuda -model ./DM_V0_transfer_4/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_4/configFiles/test/testConfig.cfg
../deepMedicRun -dev cuda -model ./DM_V0_transfer_4/output/cnnModels/trainSession/*final*.save -test ./DM_V0_transfer_4/configFiles/validation/valConfig.cfg 
