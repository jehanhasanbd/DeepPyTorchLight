<div align="center">

# 🔥 PyTorch: Zero to LLM

### *Master Deep Learning — From Tensors to Building Your Own GPT*

<br>

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Made with Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4%20-red?style=flat-square)](https://github.com/yourusername)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](http://makeapullrequest.com)
[![Stars](https://img.shields.io/github/stars/yourusername/yourrepo?style=social)](https://github.com/yourusername/yourrepo)

</div>

---

## 📌 **Navigation**
| Section | Focus |
|--------|-------|
| [1️⃣](#1️⃣-fundamentals-of-pytorch) | PyTorch basics: Tensors & Autograd |
| [2️⃣](#2️⃣-data-engineering) | Datasets & DataLoaders |
| [3️⃣](#3️⃣-training-pipeline-architecture) | Perceptron → MLP |
| [4️⃣](#4️⃣-artificial-neural-networks-ann) | Regression & Classification |
| [5️⃣](#5️⃣-computer-vision-) | CNN: Cat vs Dog |
| [6️⃣](#6️⃣-natural-language-processing-) | RNN → LSTM → Build your own GPT |

---

## 1️⃣ **Fundamentals of PyTorch**
> *The building blocks of every neural network*

| Concept | Code |
|---------|------|
| 🧩 Tensor Operations | [▶️ Launch](1_Fundamental_Of_Pytorch/1_Tensor_In_Pytorch.ipynB) |
| ⚙️ Autograd Engine | [▶️ Launch](1_Fundamental_Of_Pytorch/2_Autograd.ipynb) |

---

## 2️⃣ **Data Engineering**
> *Load, transform, and iterate like a pro*

| Component | Code |
|-----------|------|
| 📦 Dataset & DataLoader | [▶️ Launch](2_Data_Gathering/1_Dataset_And_Dataloader.ipynb) |

---

## 3️⃣ **Training Pipeline Architecture**

### 🧠 Perceptron
| Implementation | Code |
|----------------|------|
| From scratch (no nn.Module) | [▶️ Launch](3_Training_Pipeline/1_Perceptron/1_Without_NN_Module.ipynb) |
| Using nn.Module | [▶️ Launch](3_Training_Pipeline/1_Perceptron/2_Using_NN_Module.ipynb) |

### 🧬 Multi-Layer Perceptron
| Implementation | Code |
|----------------|------|
| MLP with nn.Module | [▶️ Launch](3_Training_Pipeline/2_Multi_Layer_perceptron/1_MLP_Using_NN_Module.ipynb) |
| Sequential API | [▶️ Launch](3_Training_Pipeline/2_Multi_Layer_perceptron/2_MLP_Using_Sequential_Module.ipynb) |
| With DataLoader | [▶️ Launch](3_Training_Pipeline/2_Multi_Layer_perceptron/3_Dataload_MLP_Using_Sequential_Module.ipynb) |

---

## 4️⃣ **Artificial Neural Networks (ANN)**

### 📈 Regression
| Problem | Code |
|---------|------|
| ⛽ Fuel Price Prediction | [▶️ Launch](4_Artiificial_Neural_Network/1_Regression/1_Fuel_Price_predict.ipynb) |
| 🏠 Housing Price Prediction | [▶️ Launch](4_Artiificial_Neural_Network/1_Regression/2_House_Price_Prediction.ipynb) |

### 🏷️ Classification
| Problem | Code |
|---------|------|
| 🎓 Grade Prediction (Multiclass) | [▶️ Launch](4_Artiificial_Neural_Network/2_Classication/1_Grade_prediction_multiclass.ipynb) |

---

## 5️⃣ **Computer Vision** 👁️
> *See the world through CNNs*

| Project | Code |
|---------|------|
| 🐱🐶 Cat vs Dog Classifier | [▶️ Launch](5_Computer_Vision/1_CNN_Architechture/1_Cat_Dog_Classification.ipynb) |

---

## 6️⃣ **Natural Language Processing** 📝
> *From sequences to generative AI*

### 🔁 RNN
| Task | Code |
|------|------|
| ❓ Question Answer System | [▶️ Launch](6_Natural_Language_Processing/1_Fundamental/1_RNN/1_QA_System.ipynb) |
| 🔮 Next Word Prediction | [▶️ Launch](6_Natural_Language_Processing/1_Fundamental/1_RNN/2_Next_Word_Prediction.ipynb) |

### 🧠 LSTM
| Task | Code |
|------|------|
| ❓ Question Answer System | [▶️ Launch](6_Natural_Language_Processing/1_Fundamental/2_LSTM/1_QA_System.ipynb) |
| 🔮 Next Word Prediction | [▶️ Launch](6_Natural_Language_Processing/1_Fundamental/2_LSTM/2_Next_Word_Prediction.ipynb) |

### 🔁 Bidirectional LSTM
| Task | Code |
|------|------|
| 🔮 Next Word Prediction | [▶️ Launch]() |

---

## 🏗️ **Build Your Own LLM (GPT Series)**

### 📊 Preprocessing
| Technique | Code |
|-----------|------|
| ✏️ Custom Space-Based Encoding | [▶️ Launch](6_Natural_Language_Processing/2_Build_Large_Language_Model/1_Preprocessing/1_Space_Based_Encoding.ipynb) |
| 🔡 Byte-Pair Encoding (BPE) | [▶️ Launch](6_Natural_Language_Processing/2_Build_Large_Language_Model/1_Preprocessing/2_Byte_Pair_Encoding.ipynb) |

---

### 🎯 Attention Mechanism

<details>
<summary><b>📌 Self-Attention (click to expand)</b></summary>

| # | Type | Code |
|---|------|------|
| 1 | Non-parameterized | [▶️ Launch](6_Natural_Language_Processing/2_Build_Large_Language_Model/2_Attention_Mechanism/1_Self_Attention/1_Simplified_Self_attention/1_Self_Attention_Non_parameterized_Weight_Calculation.ipynb) |
| 2 | Trainable parameterized | [▶️ Launch](6_Natural_Language_Processing/2_Build_Large_Language_Model/2_Attention_Mechanism/1_Self_Attention/1_Simplified_Self_attention/2_Self_Attention_Trainable_parameterized_Weight_Calculation.ipynb) |

</details>

<details>
<summary><b>🔒 Masked Self-Attention</b></summary>

| # | Implementation | Code |
|---|----------------|------|
| 1 | Basic masked attention | [▶️ Launch](6_Natural_Language_Processing/2_Build_Large_Language_Model/2_Attention_Mechanism/1_Self_Attention/2_Masked_Self_Attention/1_Masked_Self_Attention.ipynb) |
| 2 | Batch + Class-based | [▶️ Launch](6_Natural_Language_Processing/2_Build_Large_Language_Model/2_Attention_Mechanism/1_Self_Attention/2_Masked_Self_Attention/2_Class_Build_Masked_Self_attention.ipynb) |

</details>

<details>
<summary><b>🧠 Multi-Head Masked Attention</b></summary>

| # | Implementation | Code |
|---|----------------|------|
| 1 | Wrapper class | [▶️ Launch](6_Natural_Language_Processing/2_Build_Large_Language_Model/2_Attention_Mechanism/1_Self_Attention/3_Multi_Head_Attention/1_Wrapper_Class_Multi_Head_Attention.ipynb) |
| 2 | Optimized version | [▶️ Launch](6_Natural_Language_Processing/2_Build_Large_Language_Model/2_Attention_Mechanism/1_Self_Attention/3_Multi_Head_Attention/2_Multi_Head_Attention.ipynb) |

</details>

---

### 🧱 LLM Architecture (GPT-2 Family)
| Component | Code |
|-----------|------|
| 🏛️ GPT Model Architecture | [▶️ Launch]() |
| 📐 Layer Normalization | [▶️ Launch]() |
| ⚡ Feed Forward + GELU | [▶️ Launch]() |
| 🔗 Shortcut Connections | [▶️ Launch]() |
| 🧩 Transformer Block | [▶️ Launch]() |
| **Complete Models** | |
| 🟢 Small GPT-2 (124M) | [▶️ Launch]() |
| 🔵 Medium GPT-2 (350M) | [▶️ Launch]() |
| 🟠 Large GPT-2 (750M) | [▶️ Launch]() |
| 🔴 XL GPT-2 (1.63B) | [▶️ Launch]() |
| ✨ Text Generation | [▶️ Launch]() |

---

### ⚙️ Pretraining Pipeline
| Step | Code |
|------|------|
| 🔄 Text ↔ Token | [▶️ Launch]() |
| 📉 Loss Function | [▶️ Launch]() |
| 🏋️ Training Loop | [▶️ Launch]() |
| 🎲 Randomness Injection | [▶️ Launch]() |
| 💾 Load Pretrained Weights | [▶️ Launch]() |

---

<div align="center">

## 🚀 **Start Your Journey**

| Stage | Focus |
|-------|-------|
| 🟢 Beginner | PyTorch basics + Perceptron |
| 🟡 Intermediate | ANN + CNN + RNN/LSTM |
| 🔴 Advanced | Build GPT from scratch |

<br>

**⭐ Star this repo — You're building an LLM. Own it.**

</div>