#!/bin/bash
model_path=$(echo ./output/saved_models/trainSession/*final* | tr " " "\n" | head -n 1 | cut -d'.' -f12 --complement)
../deepMedicRun -dev cuda0 -model ./configFiles/model/modelConfig.cfg -test ./configFiles/test/testConfig.cfg -load $model_path
../deepMedicRun -dev cuda0 -model ./configFiles/model/modelConfig.cfg -test ./configFiles/validation/valConfig.cfg -load $model_path