#!/bin/bash
model_path=$(echo ../DM_VX_X/output/saved_models/trainSession/*final* | tr " " "\n" | head -n 1 | cut -d'.' -f12 --complement)
../deepMedicRun -resetopt -dev cuda0 -model ./configFiles/model/modelConfig.cfg -train ./configFiles/train/trainConfig.cfg -load $model_path