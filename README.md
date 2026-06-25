# DiffSIDRec

DiffSIDRec 是一个用于序列推荐的 diffusion-enhanced semantic-ID generative recommendation 框架。模型将下一物品预测转化为固定长度语义 ID（Semantic ID, SID）的自回归生成任务，并在不增加 SID 长度、不改变 tokenizer 结构的前提下，增强 SID code token 的连续表示能力。

本项目主要包含两个核心模块：

- **HSRM（Hierarchical Semantic Refinement Module）**：对 SID code token 的连续 embedding 进行多阶段 diffusion-style refinement、code-level projection 和动态门控融合。
- **LDB（Latent Diffusion Bridge）**：在训练阶段为 encoder-side sequence representation 和 decoder-side generation representation 提供共享 latent space 的辅助协调约束。

## 环境配置

推荐使用 Conda 环境：

```bash
conda create -n diffsidrec python=3.9
conda activate diffsidrec
pip install torch numpy pyyaml accelerate faiss-cpu tqdm scikit-learn transformers colorama
```

如果使用 GPU，请根据自己的 CUDA 版本安装对应的 PyTorch。

## 项目结构

```text
DIFFIDREC/
├── main.py                    # DiffSIDRec 主训练入口
├── model.py                   # DiffSIDRec 模型，包括 HSRM 和 LDB
├── trainer.py                 # 交替训练、验证和测试流程
├── data.py                    # 数据读取与 batch 构造
├── run.sh                     # 主模型训练脚本
├── config/
│   └── scientific.yaml        # Scientific 数据集配置文件
└── RQVAE/
    ├── main.py                # RQ-VAE 预训练入口
    ├── run_pretrain.sh        # RQ-VAE 预训练脚本
    ├── models/                # RQ-VAE 相关模块
    └── dataset/               # 数据集放在这里
```

## 数据集应该放在哪里

代码默认从下面这个目录读取数据：

```text
RQVAE/dataset/{dataset_name}/
```

当前默认配置文件是 `config/scientific.yaml`，其中：

```yaml
dataset: scientific
data_path: ./RQVAE/dataset
map_path: .emb_map.json
semantic_emb_path: scientific_emb_256.npy
rqvae_path: ./RQVAE/dataset/scientific/256-512-256-128.rqvae.pth
```

所以 Scientific 数据集需要放成下面这样：

```text
RQVAE/dataset/scientific/
├── scientific.train.jsonl
├── scientific.valid.jsonl
├── scientific.test.jsonl
├── scientific.emb_map.json
├── scientific_emb_256.npy
└── 256-512-256-128.rqvae.pth
```

代码实际识别路径如下：

- 训练集：`./RQVAE/dataset/scientific/scientific.train.jsonl`
- 验证集：`./RQVAE/dataset/scientific/scientific.valid.jsonl`
- 测试集：`./RQVAE/dataset/scientific/scientific.test.jsonl`
- item 映射表：`./RQVAE/dataset/scientific/scientific.emb_map.json`
- 语义 embedding：`./RQVAE/dataset/scientific/scientific_emb_256.npy`
- RQ-VAE checkpoint：`./RQVAE/dataset/scientific/256-512-256-128.rqvae.pth`

如果要换成其他数据集，例如 `game`，目录应写成：

```text
RQVAE/dataset/game/
├── game.train.jsonl
├── game.valid.jsonl
├── game.test.jsonl
├── game.emb_map.json
├── game_emb_256.npy
└── your_rqvae_checkpoint.pth
```

然后复制一份配置文件：

```bash
cp config/scientific.yaml config/game.yaml
```

并修改：

```yaml
dataset: game
semantic_emb_path: game_emb_256.npy
rqvae_path: ./RQVAE/dataset/game/your_rqvae_checkpoint.pth
```

## 数据文件格式

`.jsonl` 文件中每一行是一条 JSON 数据，格式如下：

```json
{"inter_history": ["item_a", "item_b", "item_c"], "target_id": "item_d"}
```

其中：

- `inter_history` 是用户历史交互序列；
- `target_id` 是需要预测的目标物品；
- 训练、验证、测试文件都使用同样的格式。

`*.emb_map.json` 用来把原始 item token 映射到整数 ID。注意：ID 应该从 1 开始，因为 0 在代码中作为 padding index。

```json
{
  "item_a": 1,
  "item_b": 2,
  "item_c": 3
}
```

`*_emb_256.npy` 是 item 的语义向量文件，默认 hidden size 为 256。它的第一维数量应与 `*.emb_map.json` 中的 item 数量对应。

## 完整运行流程

### 1. 准备数据

先确认数据已经放在：

```text
RQVAE/dataset/scientific/
```

至少需要：

```text
scientific.train.jsonl
scientific.valid.jsonl
scientific.test.jsonl
scientific.emb_map.json
scientific_emb_256.npy
```

如果你已经有预训练好的 RQ-VAE checkpoint，例如：

```text
RQVAE/dataset/scientific/256-512-256-128.rqvae.pth
```

可以直接跳到第 3 步训练 DiffSIDRec。

### 2. 预训练 RQ-VAE

RQ-VAE 用来把 item semantic embedding 转换成固定长度 semantic IDs。

在项目根目录执行：

```bash
cd RQVAE
bash run_pretrain.sh
cd ..
```

默认读取：

```text
RQVAE/dataset/scientific/scientific_emb_256.npy
```

默认保存到：

```text
RQVAE/rqvae_ckpt/scientific/
```

训练完成后，在 `config/scientific.yaml` 中把 `rqvae_path` 指向你要使用的 checkpoint，例如：

```yaml
rqvae_path: ./RQVAE/rqvae_ckpt/scientific/xxx/best_collision_model.pth
```

如果你使用的是已经准备好的 checkpoint，也可以保持：

```yaml
rqvae_path: ./RQVAE/dataset/scientific/256-512-256-128.rqvae.pth
```

### 3. 训练 DiffSIDRec

回到项目根目录执行：

```bash
bash run.sh
```

等价命令是：

```bash
accelerate launch --config_file accelerate_config_ddp.yaml main.py \
  --config ./config/scientific.yaml
```

训练过程中主要包括：

- 根据用户历史交互序列读取 item IDs；
- 使用 RQ-VAE tokenizer 为 item 生成 semantic IDs；
- 使用 T5-style encoder-decoder backbone 自回归生成目标 item 的 SID；
- 使用 HSRM 增强 SID code-token embedding；
- 使用 LDB 在训练阶段协调 encoder-side 和 decoder-side 表示；
- 按照 `cycle` 进行 tokenizer-side 和 recommender-side 的交替优化。

训练 checkpoint 默认保存到：

```text
myckpt/{dataset}/{run_name}/
```

日志默认保存到：

```text
logs/{dataset}/
```

### 4. 测试与评价

训练脚本会自动在 valid/test split 上进行评价。默认指标在 `config/scientific.yaml` 中设置：

```yaml
metrics: recall@1,recall@5,ndcg@5,recall@10,ndcg@10
valid_metric: ndcg@10
```

模型通过 beam search 生成候选 SID 序列。只有当生成的 SID 每一个 code position 都与目标 item 的 SID 完全一致时，才记为预测正确。

## 重要配置说明

主模型训练参数：

```yaml
epochs: 40
batch_size: 512
eval_batch_size: 32
lr_rec: 0.005
lr_id: 0.0001
weight_decay: 0.05
cycle: 2
early_stop: 15
```

Semantic ID 参数：

```yaml
code_num: 256
code_length: 4
```

T5-style backbone 参数：

```yaml
encoder_layers: 6
decoder_layers: 6
d_model: 128
d_ff: 512
num_heads: 4
num_beams: 20
```

RQ-VAE 参数：

```yaml
num_emb_list: [256,256,256]
e_dim: 128
layers: [512,256]
vq_type: vq
dist: l2
loss_type: mse
```

## GitHub 上传说明

`.gitignore` 已经默认忽略大文件和运行输出：

```text
RQVAE/dataset/
RQVAE/rqvae_ckpt/
myckpt/
logs/
*.pth
*.pt
*.npy
```

所以上传到 GitHub 时默认只上传代码和配置，不上传数据集、embedding、checkpoint。别人 clone 项目后，需要自己把数据放回 `RQVAE/dataset/{dataset_name}/` 才能运行。

## Citation

If you use this code, please cite the DiffSIDRec paper once the final citation information is available.
