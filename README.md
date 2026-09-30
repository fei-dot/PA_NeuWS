# Clean reconstruction

这个目录只保留真实实验重建的核心代码，结构参考 `NeuWS_DualMMES_code`：配置在文件顶部，入口按执行顺序从上往下阅读，训练循环直接写在入口中。

## 文件

```text
config.py                       常用路径和参数
data.py                         MAT读取、中心裁剪、相位重采样
model.py                        当前实际使用的静态NeuWS网络
reconstruct_full.py             完整600×600网络联合重建
reconstruct_center_wiener.py    中心相位估计 + 50帧全场Wiener
```

## 运行

在 PyCharm 中把工作目录设为本目录，然后运行：

```powershell
python reconstruct_full.py
```

或：

```powershell
python reconstruct_center_wiener.py
```

两个入口的区别：

- `reconstruct_full.py`：600×600观测和SLM相位直接进入网络；网络同时输出600×600物体和系统复场，不做尺寸转换。
- `reconstruct_center_wiener.py`：中心窗口进入网络估计中心复场，随后把复场插值到600×600，用50张调制PSF做全场Wiener。

第二条路径里，保存的 `center_field` / `center_phase` 是网络直接估计的中心结果；真正参与全场重建的是 `full_field`。每一帧都会计算
`full_field * exp(i * SLM_PHASE_SIGN * slm_phase[k])` 对应的600×600调制PSF，而不是拿中心尺寸的PSF直接卷积整幅图。

旧目录中的 `run_isoplanatic_600.py`、`main_isoplanatic.py` 和诊断脚本继续保留，用于复现历史分辨率图和等域面分析；它们不再是理解核心重建过程的必读文件。
