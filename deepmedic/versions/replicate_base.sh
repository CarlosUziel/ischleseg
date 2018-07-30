#!/bin/bash
echo DM_V0_{0..4} | xargs -n 1 cp -r DM_V0_base/*
echo DM_V0_transfer_{0..4} | xargs -n 1 cp -r DM_V0_transfer_base/*

echo DM_V0_R_{0..4} | xargs -n 1 cp -r DM_V0_R_base/*
echo DM_V0_R_transfer_{0..4} | xargs -n 1 cp -r DM_V0_R_transfer_base/*

echo DM_V1_{0..4} | xargs -n 1 cp -r DM_V1_base/*
echo DM_V1_transfer_{0..4} | xargs -n 1 cp -r DM_V1_transfer_base/*

echo DM_V1_R_{0..4} | xargs -n 1 cp -r DM_V1_R_base/*
echo DM_V1_R_transfer_{0..4} | xargs -n 1 cp -r DM_V1_R_transfer_base/*

echo DM_V2_{0..4} | xargs -n 1 cp -r DM_V2_base/*
echo DM_V2_R_{0..4} | xargs -n 1 cp -r DM_V2_R_base/*

echo DM_V3_{0..4} | xargs -n 1 cp -r DM_V3_base/*

echo DM_V4_{0..4} | xargs -n 1 cp -r DM_V4_base/*

echo DM_V5_{0..4} | xargs -n 1 cp -r DM_V5_base/*
