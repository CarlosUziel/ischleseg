#!/bin/bash
rm -rf output
../../deepMedicRun -dev cuda0 -model ./configFiles/model/modelConfig.cfg -train ./configFiles/train/trainConfig.cfg