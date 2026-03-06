
# ReMA: Recursive Multimodal Agent

[![arXiv](https://img.shields.io/badge/arXiv-2603.05484-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2603.05484)
[![hf_dataset](https://img.shields.io/badge/%F0%9F%A4%97-MM%E2%80%90Lifelong-9C276A.svg)](https://huggingface.co/datasets/CG-Bench/MM-Lifelong)


This repository contains the official implementation of the paper [Towards Multimodal Lifelong Understanding: A Dataset and Agentic Baseline](https://arxiv.org/abs/xxxx.xxxxx), which employs dynamic memory management to iteratively update a recursive belief state, significantly outperforming existing methods on lifelong context.

## News
- **2026/02/10**: Initial release of the codebase and dataset.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cg1177/Recursive-Multimodal-Agent.git
   cd Recursive-Multimodal-Agent
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Videos

#### 1.1 Download Datasets

* Download the **MM-Lifelong** dataset.
* Download **EgoLife A1–Jack** videos.

Make sure you have all video clips for each subset.

#### 1.2 Merge Video Clips

Merge all video clips **in chronological order** into a single video for each subset, and save them to the following paths:

* **stream subset**
  `./stream/merged.mp4`

* **game subset**
  `./game/merged.mp4`

* **egolife subset**
  `./egolife/merged.mp4`


### 2. Configure the Model in `run.py`

#### 2.1 Update Model Name and API Settings

Edit the following code according to your deployment:

```python
if __name__ == "__main__":
    agent = Agent(
        'stream', # stream/game/egolife
        'test',  # instance_id (any string)
        16,  # max iterations
        {
            'model_name': "YOUR_PERCEPTION_MODEL_NAME",
            'api_key': 'YOUR_PERCEPTION_MODEL_API_KEY',
            'api_base': 'YOUR_PERCEPTION_MODEL_API_BASE'
        },
        {
            'model_name': "YOUR_CONTROLLER_MODEL_NAME",
            'api_key': 'YOUR_CONTROLLER_MODEL_API_KEY',
            'api_base': 'YOUR_CONTROLLER_MODEL_API_BASE'
        }
    )
```

#### 2.2. Set Environment Variables for the Embedder

In `run.py`, set the following environment variables:

```python
os.environ["OPENAI_API_KEY"] = ""   # for embedder
os.environ["OPENAI_BASE_URL"] = ""
```

Fill in the correct API key and base URL if required by your embedder.



### 3. Run the Program

Finally, execute:

```bash
python3 run.py
```

If everything is configured correctly, the program will load the video and start running the agent. Note: on the first run, the system will perform frame caching, clip caption generation, and initial memory construction for the selected subset. This preprocessing step may take a significant amount of time.

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{chen2026multimodallifelongunderstandingdataset,
      title={Towards Multimodal Lifelong Understanding: A Dataset and Agentic Baseline}, 
      author={Guo Chen and Lidong Lu and Yicheng Liu and Liangrui Dong and Lidong Zou and Jixin Lv and Zhenquan Li and Xinyi Mao and Baoqi Pei and Shihao Wang and Zhiqi Li and Karan Sapra and Fuxiao Liu and Yin-Dong Zheng and Yifei Huang and Limin Wang and Zhiding Yu and Andrew Tao and Guilin Liu and Tong Lu},
      year={2026},
      eprint={2603.05484},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.05484}, 
}
```

## Acknowledgement

- Our codebase is partially inspired by and built upon the excellent work from [DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery).
