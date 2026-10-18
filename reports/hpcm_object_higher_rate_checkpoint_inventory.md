# Object 更高码率 HPCM checkpoint 核查

2026-10-16，在 `exp/hpcm3-object-dist` 分支完成 Object λ=0.0483 的 500-episode RGB Reconstruction 对照后，检索了本仓库 `HPCM/` 和机械盘实验目录 `/media/cjs/shared/linux/Align_adapter/` 中的 HPCM 权重文件（`*.pth.tar`、`*hpcm*.pt`、`*hpcm*.ckpt`、`*hpcm*.safetensors` 以及文件名含 `0.0` 的 `.pth*`）。结果只有：

| HPCM checkpoint | SHA256 | 文件大小 |
| --- | --- | ---: |
| `HPCM/ckpt/0.0018.pth.tar` | `dc7a6c0f4b93f2294df1ce5fc49a21ab5d8107b91c14e0ee819a9f8dbbb30964` | 274,072,529 bytes |
| `HPCM/ckpt/0.0483.pth.tar` | `d0ade22a2f24be655d441b9b52c8f4c7ce0a2a8e862bfb6f9b486f6a787686fe` | 274,072,529 bytes |

没有发现 λ>0.0483 的现存 HPCM checkpoint，因此也无法核验更高的**实际 bitstream bpp**。本阶段不假设模型存在、不虚构码率，也不启动大型 HPCM 重训。按预定顺序，下一步转为重新训练 λ=0.0018 Spatial HPCM3，并与已完成的同协议 500-episode RGB Reconstruction（485/500）配对。Object 目前仅有 λ=0.0483 一组双路径正式对照；未满足 Object 双码率曲线，最终论文图表必须如实标注缺失点，不得把单点画成曲线。

## 2026-10-18 官方发布列表复核

[HPCM 官方 GitHub 的 Pretrained Model 表](https://github.com/lyq133/LIC-HPCM#pretrained-model)列出的 **HPCM-Base / MSE** λ 依次为 `0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483`；`0.0483` 已是该系列公开的最高 λ。表中并列的 `2.4 … 60.5` 是 **MS-SSIM** 优化模型的另一套 λ 标度，不能把 `60.5` 当作高于 `0.0483` 的 MSE 模型。官方另有 HPCM-Large 和 HPCM-1B，但不是当前 HPCM-Base 架构/权重的同变量对照。本地 Object 训练与评测配置明确使用 `model_name: HPCM_Base`、MSE `0.0483.pth.tar`（SHA 同上）。因此用户提出的“从官方下载更高 λ 的 HPCM-Base 并继续原协议”目前无法执行，不能下载不对应的权重替代。

如需 Object 第二码率而保持 HPCM-Base/MSE 系列，官方可下载的备选是**较低**的 `0.025`，但这会改变原先“更高码率”的方向；是否采用应另行明确，且实际码率仍须以 Object 图像的真实 bitstream 测定，不能仅凭 λ 推断。
