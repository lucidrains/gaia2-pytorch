<img src="./gaia2.png" width="450px"></img>

## Gaia2 - Pytorch (wip)

Implementation of the [world model architecture](https://arxiv.org/abs/2503.20523) proposed for the domain of self driving out of Wayve

Please let me know your thoughts of the paper [here](https://discord.gg/na5MQBUJqb), positive or negative, so I can gauge its significance / prioritize

## Install

```bash
$ pip install gaia2-pytorch
```

## Usage

```python
import torch
from gaia2_pytorch import VideoTokenizer, Gaia2

video = torch.randn(1, 3, 10, 16, 16)

tokenizer = VideoTokenizer()

loss = tokenizer(video)
loss.backward()

gaia2 = Gaia2(tokenizer)

loss = gaia2(video)
loss.backward()

generated = gaia2.generate((10, 16, 16))
assert generated.shape == video.shape
```

## Contributing

```bash
$ pip install '.[test]'
```

Then add a test to `tests` and run the following

```bash
$ pytest tests
```

That's it

## Citations

```bibtex
@article{Russell2025GAIA2AC,
    title   = {GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving},
    author  = {Lloyd Russell and Anthony Hu and Lorenzo Bertoni and George Fedoseev and Jamie Shotton and Elahe Arani and Gianluca Corrado},
    journal = {ArXiv},
    year    = {2025},
    volume  = {abs/2503.20523},
    url     = {https://api.semanticscholar.org/CorpusID:277321454}
}
```

```bibtex
@article{Rombach2021HighResolutionIS,
    title   = {High-Resolution Image Synthesis with Latent Diffusion Models},
    author  = {Robin Rombach and A. Blattmann and Dominik Lorenz and Patrick Esser and Bj{\"o}rn Ommer},
    journal = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2021},
    pages   = {10674-10685},
    url     = {https://api.semanticscholar.org/CorpusID:245335280}
}
```

```bibtex
@article{Zhu2025FracConnectionsFE,
    title   = {Frac-Connections: Fractional Extension of Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Jundong Zhou and Zihao Huang and Yutao Zeng and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2025},
    volume  = {abs/2503.14125},
    url     = {https://api.semanticscholar.org/CorpusID:277104144}
}
```
