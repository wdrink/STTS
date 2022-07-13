# [Efficient Video Transformers with Spatial-Temporal Token Selection](https://arxiv.org/abs/2111.11591)

Official PyTorch implementation of STTS, from the following paper:

[Efficient Video Transformers with Spatial-Temporal Token Selection](https://arxiv.org/abs/2111.11591), ECCV 2022.

[Junke Wang<sup>*</sup>](https://www.wangjunke.info/),[Xitong Yang<sup>*</sup>](http://www.xyang35.umiacs.io/), [Hengduo Li](https://henrylee2570.github.io/), Li Liu, [Zuxuan Wu](https://zxwu.azurewebsites.net/), [Yu-Gang Jiang](http://www.yugangjiang.info/).

Fudan University, University of Maryland, BirenTech Research

--- 

<p align="center">
<img src="./imgs/teaser.png" width=100% height=100% 
class="center">
</p>

We present STTS, a token selection framework that dynamically selects a few informative tokens in both temporal and spatial dimensions conditioned on input video samples.


## Model Zoo

### [MViT](https://arxiv.org/abs/2104.11227) with STTS on Kinetics-400

| name | acc@1 | FLOPs | model |
|:---:|:---:|:---:|:---:|
| MViT-T<sup>0</sup><sub>0.9</sub>-S<sup>4</sup><sub>0.9</sub> | 78.1 | 56.4 | [model](https://drive.google.com/file/d/1IP_phCBQRTsUb5RQmPeRusjdlM2J10Xb/view?usp=sharing) |
| MViT-T<sup>0</sup><sub>0.8</sub>-S<sup>4</sup><sub>0.9</sub> | 77.9 | 47.2 | [model](https://drive.google.com/file/d/1Fd0q3e9VDfokfuljt0hF7FXfRxh3QO61/view?usp=sharing) |
| MViT-T<sup>0</sup><sub>0.6</sub>-S<sup>4</sup><sub>0.9</sub> | 77.5 | 38.1 | [model](https://drive.google.com/file/d/1Zkn-AY6Pb2wMuzQrUFXQYGJLugVG9BQF/view?usp=sharing) |
| MViT-T<sup>0</sup><sub>0.5</sub>-S<sup>4</sup><sub>0.7</sub> | 76.6 | 23.3 | [model](https://drive.google.com/file/d/1JSESMIOi1A-9QaQflcoxCPS7RYxa03pH/view?usp=sharing) |
| MViT-T<sup>0</sup><sub>0.4</sub>-S<sup>4</sup><sub>0.6</sub> | 75.6 | 12.1 | [model](https://drive.google.com/file/d/1A5qA1d6lIpskV8cQVjDFuwD128SI4OfJ/view?usp=sharing) |

### [VideoSwin](https://arxiv.org/abs/2106.13230) with STTS on Kinetics-400

| name | acc@1 | FLOPs | model |
|:---:|:---:|:---:|:---:|
| VideoSwin-T<sup>0</sup><sub>0.9</sub> | 81.9 | 252.5 | [model](https://drive.google.com/file/d/1GhIUgFBLTBQqZZ0puEtTSJPB_4ca77oy/view?usp=sharing) |
| VideoSwin-T<sup>0</sup><sub>0.8</sub> | 81.6 | 223.4 | [model](https://drive.google.com/file/d/1gqLCuDUeSYjx2XKqPPaZDDWD9WDWnJjb/view?usp=sharing) |
| VideoSwin-T<sup>0</sup><sub>0.6</sub> | 81.4 | 181.4 | [model](https://drive.google.com/file/d/1bGIULKgISA5EGanO1Es6_tvqvQsoAnrb/view?usp=sharing) |
| VideoSwin-T<sup>0</sup><sub>0.5</sub> | 81.1 | 121.6 | [model](https://drive.google.com/file/d/1xokwKomvCkGfyATaY_2yzjQr9OcBi995/view?usp=sharing) |
| VideoSwin-T<sup>0</sup><sub>0.4</sub> | 80.7 | 91.4 | [model](https://drive.google.com/file/d/1W6sQp7bjKDWleARivkuoMmZOWzi5YKsl/view?usp=sharing) |


## Installation
Please check [MViT](https://github.com/facebookresearch/SlowFast) and [VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer) for installation instructions. 

## Training and Evaluation

### MViT

For both training and evaluation with MViT as backbone, you could use:

```
cd MViT

python tools/run_net.py --cfg path_to_your_config
```

For example, to train MViT-T<sup>0</sup><sub>0.6</sub>-S<sup>4</sup><sub>0.9</sub>, run:

```
python tools/run_net.py --cfg configs/Kinetics/t0_0.6_s4_0.9.yaml
```


### VideoSwin

For training, you could use:

```
cd VideoSwin

bash tools/dist_train.sh path_to_your_config $NUM_GPUS --checkpoint path_to_your_checkpoint --validate --test-last
```

while for evaluation, you could use:

```
bash tools/dist_test.sh path_to_your_config path_to_your_checkpoint $NUM_GPUS --eval top_k_accuracy

```

For example, to evaluate VideoSwin-T<sup>0</sup><sub>0.9</sub> on a single node with 8 gpus, run:

```
cd VideoSwin

bash tools/dist_test.sh configs/Kinetics/t0_0.875.py ./checkpoints/t0_0.875.pth 8 --eval top_k_accuracy
```

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{wang2021efficient,
  title={Efficient video transformers with spatial-temporal token selection},
  author={Wang, Junke and Yang, Xitong and Li, Hengduo and Li, Liu and Wu, Zuxuan and Jiang, Yu-Gang},
  booktitle={ECCV},
  year={2022}
}
```
