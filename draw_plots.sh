#!/bin/bash

python ./deepmedic/plotTrainingProgress.py ../milestones/DM_V1/output_DM_V1/logs/trainSession_DM_V1.txt -d -c 0

python ./deepmedic/plotTrainingProgress.py ../milestones/DM_V1/output_DM_V1_transfer/logs/trainSession_DM_V1_transfer.txt -d -c 0

python ./deepmedic/plotTrainingProgress.py ../milestones/DM_V2/output_DM_V2/logs/trainSession_DM_V2.txt -d -c 0

python ./deepmedic/plotTrainingProgress.py ../milestones/DM_V2/output_DM_V2_transfer/logs/trainSession_DM_V2_transfer.txt -d -c 0
