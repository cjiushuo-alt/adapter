# Object 更高码率 HPCM checkpoint 核查

2026-10-16，在 `exp/hpcm3-object-dist` 分支完成 Object λ=0.0483 的 500-episode RGB Reconstruction 对照后，检索了本仓库 `HPCM/` 和机械盘实验目录 `/media/cjs/shared/linux/Align_adapter/` 中的 HPCM 权重文件（`*.pth.tar`、`*hpcm*.pt`、`*hpcm*.ckpt`、`*hpcm*.safetensors` 以及文件名含 `0.0` 的 `.pth*`）。结果只有：

| HPCM checkpoint | SHA256 | 文件大小 |
| --- | --- | ---: |
| `HPCM/ckpt/0.0018.pth.tar` | `dc7a6c0f4b93f2294df1ce5fc49a21ab5d8107b91c14e0ee819a9f8dbbb30964` | 274,072,529 bytes |
| `HPCM/ckpt/0.0483.pth.tar` | `d0ade22a2f24be655d441b9b52c8f4c7ce0a2a8e862bfb6f9b486f6a787686fe` | 274,072,529 bytes |

没有发现 λ>0.0483 的现存 HPCM checkpoint，因此也无法核验更高的**实际 bitstream bpp**。本阶段不假设模型存在、不虚构码率，也不启动大型 HPCM 重训。按预定顺序，下一步转为重新训练 λ=0.0018 Spatial HPCM3，并与已完成的同协议 500-episode RGB Reconstruction（485/500）配对。Object 目前仅有 λ=0.0483 一组双路径正式对照；未满足 Object 双码率曲线，最终论文图表必须如实标注缺失点，不得把单点画成曲线。
