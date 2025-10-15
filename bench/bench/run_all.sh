#!/bin/bash

echo "========= Starting Video Quality Assessment Pipeline ========="

echo "1. Running Dynamic Degree Analysis..."
python dynamic_degree.py

echo "2. Running Motion Smoothness Analysis..."
python motion_smoothness.py

echo "3. Running Objective Quality Analysis..."
python objective_quality.py

echo "4. Running Prompt Alignment Analysis..."
python prompt_alignment.py

echo "5. Running Scene Consistency Analysis..."
python scene_consistency.py

echo "6. Running Temporal Consistency Analysis..."
python temporal_consistency.py

echo "========= All analyses completed! ========="