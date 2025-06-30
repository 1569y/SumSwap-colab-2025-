# SumSwap-colab-2025
针对SimSwap源代码
> 这是一个面向 Colab 环境进行自动化运行的 SimSwap 视频换脸项目。
该项目来源于开源项目 [SimSwap](https://github.com/neuralchen/SimSwap)，针对其中的colab进行一些优化（兼容 PyTorch 新版本的代码补丁等），实现 Colab 环境下的完整部署、本地文件上传支持、自动处理兼容性问题以及视频换脸结果下载的全链路流程。
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/1569y/SumSwap-colab-2025-/blob/main/main.ipynb)

## 原项目链接

* GitHub 地址：[https://github.com/neuralchen/SimSwap](https://github.com/neuralchen/SimSwap)
* 纸版链接：[https://arxiv.org/abs/2106.06340](https://arxiv.org/abs/2106.06340)

## 项目特色

* 支持 **本地上传** ArcFace / Antelope / Parsing / 512 等全部模型，也可以通过云盘挂载
* 自动解压和目录修复，避免路径异常
* 一键校验模型加载状态，自动识别图像+视频
* 自动解决 PyTorch 2.6+ 与 NumPy 1.24+ 的兼容性问题
* 自动下载最新换脱视频结果

## 📃 文件的结构

```
SimSwap/
├── arcface_model/                  # ArcFace 模型 (.tar)
├── insightface_func/models/antelope/  # antelope 模型 (.onnx)
├── checkpoints/                    # SimSwap 主组件模型
├── parsing_model/checkpoint/        # 79999_iter.pth 视频分割模型
├── data/                          # 上传的视频+图像 zip 解压后文件夹
└── output/                        # 结果视频输出 (.mp4)
```

## 📚 使用步骤

1. 执行前 3 步，完成环境部署
2. 下载相应的预训练模型：
   - 原作者的准备指导: https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md
   - [antelope.zip] https://gitcode.com/Universal-Tool/f6dce/?utm_source=article_gitcode_universal&index=top&type=card&&isLogin=1
3. 按顺序上传模型文件：
   * `arcface_checkpoint.tar`
   * `antelope.zip`（含 .onnx 文件）
   * 用户端上传 checkpoints 和 parsing 模型 (optional)
4. 上传一个包含图片+视频的 zip 文件
5. 执行最后的 swap 模块，得到结果
6. 结果视频会自动下载

## 注意

* 所有模型都支持本地上传，无需 Google Drive（也有相应的代码支持 Google Drive ）
* 上传视频需包含角色头像 + 视频，格式支持 JPG/PNG + MP4
* 只支持查找 zip 文件解压后的第一张图 + 第一个视频
