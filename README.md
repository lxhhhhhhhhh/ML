### ponet.py 和 detr.py

ponet.py 中的 neoRIC11 是第二次作业中训练的残差网络，用于提取图片特征

detr.py 实现了 DETR 的 transformer 模块


### 用 ponet.neoRIC11 作为 DETR backbone

```python
import ponet
checkpoint = torch.load(<pathOfCheckpoint>)
backbone = ponet.neoRIC11()
backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
...
import detr
model = detr.DETR(backbone = backbone, num_classes = ...) # number of classes in training data
```

如果ponet中的网络表现欠佳，也可以自由加载其他预训练网络作为backbone

### 参考资料
[DETR 论文](https://ai.meta.com/research/publications/end-to-end-object-detection-with-transformers/)

[DETR GitHub repo](https://github.com/facebookresearch/detr?tab=readme-ov-file)

[DETR HF model doc](https://github.com/facebookresearch/detr?tab=readme-ov-file)