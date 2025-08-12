# 🚀 GPT大模型完整学习与实践项目

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

本项目是一个从 **零基础构建 GPT 大语言模型** 到 **多种微调策略** 再到 **实际 Web 部署** 的完整学习与实践项目。  
项目认真按照 [Build-A-Large-Language-Model-CN](https://github.com/skindhu/Build-A-Large-Language-Model-CN/tree/main) 的学习路线实现，并在此基础上进行了功能扩展与工程化改造。  
特别感谢该项目作者提供了系统且易懂的学习资源 🙏。

---

## 📌 项目特色

- **🏗 从零构建 GPT** - 纯手写实现 Transformer 核心组件，深入理解原理  
- **🔧 多种微调策略** - 分类微调、监督指令微调、LoRA 高效微调  
- **💻 一键 Web 部署** - 集成 AI 聊天助手与垃圾短信分类器  
- **⚡ GPU 加速支持** - 自动检测 CUDA，显著提升推理速度  
- **📊 完整学习路径** - 覆盖理论、实践、部署全流程  

---

## 🚀 快速开始

### 🎯 方式一：直接体验（需先训练模型）

```bash
# 1. 克隆项目
git clone https://github.com/anonymous-SBP-RVD/wangsir.git
cd wangsir

# 2. 安装依赖
pip install streamlit torch tiktoken pandas plotly

# 3. 训练所需模型（详见下方"重要提醒"）
# 4. 一键启动 Web 应用
cd deployWebapp
python autodeploy.py
```

### 🧪 方式二：学习训练过程

```bash
# 1. 进入任意训练项目
cd project/Fine-Tuning_Classifier  # 或其他项目目录

# 2. 查看具体训练说明
cat README.md  # 每个项目都有详细说明


**🎮 应用端口：**
* 🤖 AI 聊天助手：`8502`
* 🛡 垃圾短信分类器：`8501`

---

## 📂 项目结构

```
wangsir/
├── deployWebapp/           # 🌐 Web 应用部署
│   ├── autodeploy.py       # 一键启动脚本
│   ├── chat_assistant_app.py   # AI 聊天助手
│   ├── spam_classifier_app.py  # 垃圾短信分类器
│   ├── gptMoudel.py        # GPT 模型架构
│   ├── generateSimple_context.py # 文本生成工具
│   └── __pycache__/        
│
├── project/                # 🧠 核心训练实验
│   ├── Fine-Tuning_Classifier/ # 分类微调（生成 review_classifier.pth）
│   ├── LoRA-FT_Classifier/     # LoRA 微调（生成 LoRA_classifier.pth）
│   ├── SFT/                    # 监督指令微调（生成 gpt2-medium355M-sft.pth）
│   ├── zero to one/            # 从零构建 GPT
│   └── __pycache__/
│
├── gpt2/                    # 📦 预训练模型
│   ├── 124M/                 # GPT-2 Small
│   └── 355M/                 # GPT-2 Medium
│
├── sms_spam_collection/      # 📊 数据集
│   ├── SMSSpamCollection.tsv # 原始数据
│   └── readme
│
└── sms_spam_collection.zip   # 数据集压缩包
```

---

## 🎮 应用功能

### 🤖 AI 聊天助手

* 多人格选择（友善、专业、创意、导师）
* 实时调节 Temperature、Top-K
* 对话历史保存与清除
* GPU 加速推理

### 🛡 垃圾短信分类器

* 实时短信检测
* 置信度评分与风险等级提示
* 历史检测统计与可视化

---

## 🧠 技术实现

* **基础模型**：GPT-2（124M / 355M 参数）
* **核心机制**：多头自注意力、位置编码、层归一化
* **微调方法**：LoRA、监督指令微调、分类头微调
* **优化策略**：CUDA 加速、智能采样、重复抑制

---

## ⚠️ 重要提醒

由于 GitHub 文件大小限制（单文件最大 100MB），以下大模型文件未包含在此仓库中：

- 模型权重文件（`*.pth`）
- 模型检查点文件（`*.ckpt`）
- 预训练模型数据（`model.ckpt.data-*`）

**🔥 如需完整运行 Web 应用，请先训练模型：**

### 1️⃣ 训练分类器模型
```bash
# 训练垃圾短信分类器
cd project/Fine-Tuning_Classifier
python init_model.py

# 或训练 LoRA 微调版本
cd project/LoRA-FT_Classifier
python LoRA-FT-model.py
```

### 2️⃣ 训练聊天模型
```bash
# 监督指令微调
cd project/SFT
python SFT.py
```

### 3️⃣ 启动 Web 应用
```bash
# 训练完成后，将模型文件复制到 deployWebapp/ 目录
# 然后启动应用
cd deployWebapp
python autodeploy.py
```

**📝 说明**：每个 `project/` 子目录都包含完整的训练代码和说明，可以独立运行训练过程。

---

## 📄 环境要求

* **最低配置**：

  * Python 3.8+
  * RAM: 8GB+
  * 磁盘空间: 5GB+

* **推荐配置**：

  * Python 3.9+
  * RAM: 16GB+
  * GPU: NVIDIA GTX 1060+（支持 CUDA）
  * 磁盘空间: 10GB+

---

## 🙏 致谢

* [Build-A-Large-Language-Model-CN](https://github.com/skindhu/Build-A-Large-Language-Model-CN/tree/main) - 提供了本项目的核心学习框架
* [OpenAI](https://openai.com/) - GPT 架构与理论
* [Hugging Face](https://huggingface.co/) - Transformers 生态
* [Streamlit](https://streamlit.io/) - 快速构建 Web 应用


---

**⭐ 如果本项目对你有帮助，请点 Star 支持我们！**
