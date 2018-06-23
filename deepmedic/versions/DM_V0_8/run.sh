#!/bin/bash

rm -rf output
../../deepMedicRun -dev cuda -newModel ./configFiles/model/modelConfig.cfg -train ./configFiles/train/trainConfig.cfg
