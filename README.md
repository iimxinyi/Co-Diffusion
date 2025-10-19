# Co-Diffusion

## 2 Quick Start for Bench (Video Evaluation Benchmark)

**Step 1: Installation**


``` sh
conda create -n bench python=3.8
conda activate bench
cd bench
pip install -r requirements_bench.txt
```

**Step 2: Third-Party Model Download**

Download the third-party model parameters according to the instructions in the `.md` file under `bench/pretrained`.

**Step 3: Run Video Evaluation Benchmark**

For a specific metric, run:

```
python ./bench/prompt_alignment.py
python ./bench/scene_consistency.py
python ./bench/dynamic_degree.py
python ./bench/motion_smoothness.py
python ./bench/temporal_consistency.py
python ./bench/objective_quality.py
```

To run all metrics at once, run:

```
bash run_all.sh
```


