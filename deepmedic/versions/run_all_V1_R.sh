#!/bin/bash



rm -rf ./DM_V1_R_0/output
../deepMedicRun -dev cuda0 -model ./DM_V1_R_0/configFiles/model/modelConfig.cfg -train ./DM_V1_R_0/configFiles/train/trainConfig.cfg
model_path=$(echo ./DM_V1_R_0/output/saved_models/trainSession/*final* | tr " " "\n" | head -n 1 | cut -d'.' -f12 --complement)
../deepMedicRun -dev cuda0 -model ./DM_V1_R_0/configFiles/model/modelConfig.cfg -test ./DM_V1_R_0/configFiles/test/testConfig.cfg -load $model_path
../deepMedicRun -dev cuda0 -model ./DM_V1_R_0/configFiles/model/modelConfig.cfg -test ./DM_V1_R_0/configFiles/validation/valConfig.cfg -load $model_path

rm -rf ./DM_V1_R_1/output
../deepMedicRun -dev cuda0 -model ./DM_V1_R_1/configFiles/model/modelConfig.cfg -train ./DM_V1_R_1/configFiles/train/trainConfig.cfg
model_path=$(echo ./DM_V1_R_1/output/saved_models/trainSession/*final* | tr " " "\n" | head -n 1 | cut -d'.' -f12 --complement)
../deepMedicRun -dev cuda0 -model ./DM_V1_R_1/configFiles/model/modelConfig.cfg -test ./DM_V1_R_1/configFiles/test/testConfig.cfg -load $model_path
../deepMedicRun -dev cuda0 -model ./DM_V1_R_1/configFiles/model/modelConfig.cfg -test ./DM_V1_R_1/configFiles/validation/valConfig.cfg -load $model_path

rm -rf ./DM_V1_R_2/output
../deepMedicRun -dev cuda0 -model ./DM_V1_R_2/configFiles/model/modelConfig.cfg -train ./DM_V1_R_2/configFiles/train/trainConfig.cfg
model_path=$(echo ./DM_V1_R_2/output/saved_models/trainSession/*final* | tr " " "\n" | head -n 1 | cut -d'.' -f12 --complement)
../deepMedicRun -dev cuda0 -model ./DM_V1_R_2/configFiles/model/modelConfig.cfg -test ./DM_V1_R_2/configFiles/test/testConfig.cfg -load $model_path
../deepMedicRun -dev cuda0 -model ./DM_V1_R_2/configFiles/model/modelConfig.cfg -test ./DM_V1_R_2/configFiles/validation/valConfig.cfg -load $model_path

rm -rf ./DM_V1_R_3/output
../deepMedicRun -dev cuda0 -model ./DM_V1_R_3/configFiles/model/modelConfig.cfg -train ./DM_V1_R_3/configFiles/train/trainConfig.cfg
model_path=$(echo ./DM_V1_R_3/output/saved_models/trainSession/*final* | tr " " "\n" | head -n 1 | cut -d'.' -f12 --complement)
../deepMedicRun -dev cuda0 -model ./DM_V1_R_3/configFiles/model/modelConfig.cfg -test ./DM_V1_R_3/configFiles/test/testConfig.cfg -load $model_path
../deepMedicRun -dev cuda0 -model ./DM_V1_R_3/configFiles/model/modelConfig.cfg -test ./DM_V1_R_3/configFiles/validation/valConfig.cfg -load $model_path

rm -rf ./DM_V1_R_4/output
../deepMedicRun -dev cuda0 -model ./DM_V1_R_4/configFiles/model/modelConfig.cfg -train ./DM_V1_R_4/configFiles/train/trainConfig.cfg
model_path=$(echo ./DM_V1_R_4/output/saved_models/trainSession/*final* | tr " " "\n" | head -n 1 | cut -d'.' -f12 --complement)
../deepMedicRun -dev cuda0 -model ./DM_V1_R_4/configFiles/model/modelConfig.cfg -test ./DM_V1_R_4/configFiles/test/testConfig.cfg -load $model_path
../deepMedicRun -dev cuda0 -model ./DM_V1_R_4/configFiles/model/modelConfig.cfg -test ./DM_V1_R_4/configFiles/validation/valConfig.cfg -load $model_path
