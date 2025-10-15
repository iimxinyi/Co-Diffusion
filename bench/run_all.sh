#!/bin/bash

echo "========= Starting Video Quality Assessment Pipeline ========="

echo "1. Running Dynamic Degree Analysis..."
python ./bench/dynamic_degree.py

echo "2. Running Motion Smoothness Analysis..."
python ./bench/motion_smoothness.py

echo "3. Running Objective Quality Analysis..."
python ./bench/objective_quality.py

echo "4. Running Prompt Alignment Analysis..."
python ./bench/prompt_alignment.py

echo "5. Running Scene Consistency Analysis..."
python ./bench/scene_consistency.py

echo "6. Running Temporal Consistency Analysis..."
python ./bench/temporal_consistency.py

echo "========= All analyses completed! ========="