./deepMedicRun -model ./examples/configFiles/tinyCnn/model/modelConfig.cfg \
               -train examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg

python plotTrainingProgress.py examples/output/logs/trainSessionWithValidTiny.txt -d

model_path=$(echo ./examples/output/saved_models/trainSessionWithValidTiny/*final* | tr " " "\n" | head -n 1 | cut -d'.' -f12 --complement)

./deepMedicRun -model ./examples/configFiles/tinyCnn/model/modelConfig.cfg \
               -test ./examples/configFiles/tinyCnn/test/testConfig.cfg \
               -load $model_path

