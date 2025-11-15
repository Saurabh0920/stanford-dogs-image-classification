# **Stanford Dogs – Comparative Study of Six Deep Learning Approaches for Image Classification**

## **1. Introduction**

This project presents a comparative analysis of six deep learning methods for image classification using the **Stanford Dogs Dataset**, a fine-grained image recognition benchmark containing **20,580 images across 120 dog breeds**.
The objective is to evaluate how different convolutional neural network (CNN) strategies perform under identical conditions and to benchmark traditional CNNs against advanced transfer learning models.

This work is completed as part of the **M-Tech in Artificial Intelligence – Deep Learning Module (AI09 Batch)** under REVA University.

---

## **2. Objectives**

The primary objectives of this project are:

* To implement and evaluate **six distinct image-classification pipelines**.
* To compare traditional CNN models with **state-of-the-art pretrained models** (VGG16, ResNet50).
* To measure the impact of **augmentation, feature extraction, and fine-tuning**.
* To produce a consolidated performance benchmark for academic and research understanding.

---

## **3. Dataset Description**

* **Name:** Stanford Dogs Dataset
* **Source:** Kaggle ([https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset))
* **Images:** 20,580
* **Classes:** 120 dog breeds
* **Characteristics:**

  * Fine-grained classification
  * High intra-class variation
  * Complex backgrounds
  * Varying image resolutions

The dataset is **not included** in the repository due to size constraints. Instructions to download and structure the dataset are available under `data/README.md`.

---

## **4. Methods Implemented**

Six approaches were implemented to systematically measure performance differences.

### **4.1 Basic ConvNet (Scratch, No Augmentation)**

* A simple CNN architecture trained from scratch
* No transfer learning
* No image augmentation

### **4.2 Scratch CNN + Data Augmentation**

* Same scratch model with augmentation (rotation, zoom, flips, shear)
* Tests augmentation impact on low-capacity architectures

### **4.3 VGG16 – Feature Extraction**

* Pretrained VGG16 on ImageNet
* Frozen convolutional base
* Custom dense classifier trained on extracted features

### **4.4 VGG16 – Feature Extraction + Augmentation**

* Combines feature extraction with augmentation
* Tests whether augmentation interacts positively with transfer learning

### **4.5 VGG16 – Fine-Tuning (Top Layers)**

* Unfreezes a subset of top convolutional blocks
* Low learning rate for careful fine-tuning
* High risk of overfitting due to dataset size

### **4.6 ResNet50 – Feature Extraction**

* Pretrained ResNet50 (ImageNet)
* Frozen base + custom classifier
* Known for superior feature representations and deeper architecture

---

## **5. Experimental Setup**

* **Framework:** TensorFlow/Keras
* **Hardware:** GPU-enabled environment recommended
* **Train–Validation Split:** 80% / 20%
* **Optimiser:** Adam (various learning rates depending on method)
* **Image Size:** 224 × 224
* **Loss Function:** Categorical Cross-Entropy
* **Evaluation Metric:** Top-1 Validation Accuracy

Each method was implemented in a separate Jupyter Notebook (available in the `src/` directory).

---

## **6. Results & Analysis**

### **6.1 Quantitative Results**

A consolidated comparison of the six methods is presented below:

| Method                                  | Top-1 Accuracy |
| --------------------------------------- | -------------- |
| **1) Basic CNN (scratch, no aug)**      | 2.14%          |
| **2) Scratch + Data Augmentation**      | 2.66%          |
| **3) VGG16 – Feature Extraction**       | 65.84%         |
| **4) VGG16 – Feature Extraction + Aug** | 38.62%         |
| **5) VGG16 – Fine-Tuning (top layers)** | 39.72%         |
| **6) ResNet50 – Feature Extraction**    | **76.77%**     |

### **6.2 Key Observations**

* **Transfer learning dramatically outperforms scratch models**

  * Scratch models fail due to insufficient data and limited model capacity.
  * VGG16 and ResNet50 show **30–35× higher accuracy**.

* **VGG16 + Aug performed worse than pure VGG16 features**

  * Indicates augmentation pipeline interference with batch statistics
  * Augmentation beneficial only when training CNNs from scratch.

* **ResNet50 delivered the highest accuracy (76.77%)**

  * Deeper architecture and residual connections enable stronger feature extraction.

* **Fine-tuning VGG16 did not improve performance**

  * Likely causes:

    * LR too high/low
    * Overfitting of pretrained layers
    * Limited dataset size for fine-tuning

### **6.3 Results Visualization**

The performance comparison chart is available in `results/stanford_dogs_results_summary.png`:


---

## **7. Conclusion**

This study highlights the stark contrast between training CNNs from scratch and leveraging pretrained models. Key conclusions:

* **Transfer learning is essential for small and fine-grained datasets.**
* **ResNet50 is superior to VGG16** in feature extraction tasks for this dataset.
* **Fine-tuning requires careful hyperparameter tuning** and may not always improve performance.

The project provides a robust benchmark and demonstrates best practices in deep learning experimentation.

---

## **8. Repository Structure**

```
├── src/                 # Six Jupyter notebooks
├── data/                # Dataset instructions
├── docs/                # Project report & diagrams
├── results/             # CSV, plots, metrics
└── README.md
```

---

## **9. How to Run the Notebooks**

### **Step 1 — Install Dependencies**

```
pip install -r requirements.txt
```

### **Step 2 — Download Dataset**

Follow instructions in `data/README.md`.

### **Step 3 — Run Notebooks**

Start with any notebook under `src/`.

GPU is strongly recommended for pretrained models.

---

## **10. Future Enhancements**

* Full fine-tuning of ResNet50 or MobileNetV3
* Using EfficientNet or ConvNeXt
* Implementation of mixed precision training
* Hyperparameter tuning with Keras Tuner
* Convert model to ONNX / TensorRT for deployment

---

## **11. Acknowledgements**

* Stanford Dogs Dataset (Khosla et al., Stanford Vision Lab)
* ImageNet Pretrained Models (VGG16, ResNet50)
* TensorFlow/Keras Framework

next**.
