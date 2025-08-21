# 🛡️ Network Intrusion Detection System (IDS) - Machine Learning from Scratch

### **A custom-built neural network for identifying cyber threats with 88.8% accuracy**

This project presents a complete **Intrusion Detection System (IDS)** developed from the ground up using fundamental Python libraries. The system analyzes network traffic to automatically classify connections as either benign or malicious, achieving **88.8% accuracy** on real-world cybersecurity data—exceeding academic requirements while demonstrating mastery of machine learning fundamentals.

> **🏆 Academic Excellence:** Developed for a Master's-level AI coursework with strict constraints: **implement ML algorithms without high-level libraries** like Scikit-learn or TensorFlow. The project earned a grade of **85/100**, praised for its technical execution and exceeding the 85% accuracy requirement.

## ✨ Key Achievements

- **High Accuracy:** Achieved **88.8% detection accuracy** on unseen test data, surpassing the 85% requirement
- **Custom Neural Network:** Built a Multi-Layer Perceptron (MLP) completely from scratch using only NumPy
- **Real-World Data:** Trained and tested on the UNSW-NB15 dataset, a modern benchmark for network intrusion detection
- **Robust Preprocessing:** Engineered a complete data pipeline handling both numerical and categorical network data
- **Production-Ready Metrics:** Maintained low false alarm rate (6.1%) while detecting 86.6% of all attacks

## 📊 Performance Summary

**Model achieved excellent results on Testing Set 1:**
```
Accuracy:   88.80%   (Exceeds 85% requirement)
Precision:  96.40%   (Very low false alarms)
Recall:     86.59%   (Detects most attacks)
F1 Score:   91.23%   (Strong balance)
False Alarm Rate: 6.10% (Only 87 benign flagged incorrectly)
```

**Confusion Matrix:**
```
         Predicted Benign  Predicted Malicious
Actual Benign     1221               87
Actual Malicious   361              2331
```

## 🛠️ Technical Implementation

### The Toolkit
The project uses only foundational scientific computing libraries:
```bash
# No high-level ML libraries used - implemented everything from scratch
import numpy as np
import pandas as pd
import math
from collections import Counter
```

### Architecture Overview

1. **Data Preprocessing Pipeline**
   - **Categorical Encoding:** Converted protocol, service, and state features into numerical values
   - **Min-Max Scaling:** Normalized all numerical features to [0, 1] range
   - **Data Integrity:** Strict separation between training and test preprocessing to prevent leakage

2. **Custom Neural Network (MLP)**
   - **Input Layer:** 43 features (network traffic characteristics)
   - **Hidden Layers:** 64 → 32 neurons with LeakyReLU activation,output uses Sigmoid
   - **Output Layer:** 1 neuron with sigmoid activation for binary classification
   - **Training:** Mini-batch gradient descent with binary cross-entropy loss

3. **Model Training**
   - **Epochs:** 50 training cycles
   - **Batch Size:** 64 samples per update
   - **Learning Rate:** 0.01 for stable convergence
   - **Optimization:** Custom backpropagation implementation

```python
# Simplified model architecture
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        # He initialization for weights, zeros for biases
        self.weights = [np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2./input_size),
                       np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2./hidden_sizes[0]),
                       np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2./hidden_sizes[1])]
        self.biases = [np.zeros((1, hidden_sizes[0])), 
                      np.zeros((1, hidden_sizes[1])), 
                      np.zeros((1, output_size))]
```

## 📈 Results Analysis

### Testing Performance
The model was rigorously evaluated on two testing sets:

1. **Testing Set 1 (4,000 samples):** 88.8% accuracy with strong precision (96.4%)
2. **Testing Set 2 (25 samples):** Successfully generated predictions for unlabeled data

### Strengths
- **Low False Alarms:** Only 6.1% of benign traffic incorrectly flagged as malicious meaning the system rarely interrupts legitimate network traffic, a crucial requirement for real-world deployment.
- **High Detection Rate:** Identified 86.6% of all malicious attacks
- **Computational Efficiency:** Processes 4,000 samples in under 1 second on CPU

### Improvement Areas
- **Recall Optimization:** 361 malicious connections were missed (focus for future work)
- **Class Imbalance:** Some attack types were underrepresented in training data

## 🔮 Future Enhancements

1. **Advanced Architecture**
   - Implement batch normalization for faster training
   - Experiment with Swish activation functions
   - Add dropout layers for better regularization

2. **Data Optimization**
   - Apply SMOTE for class imbalance mitigation
   - Develop time-based feature engineering
   - Create interaction features between packets and services

3. **Deployment Ready**
   - Implement online learning for evolving threats
   - Create mistake-tracking system for continuous improvement
   - Develop real-time monitoring capabilities

## 👨‍💻 Author

**Lokesh Yuvraj Khadke** | MSc Artificial Intelligence

I am a data scientist and machine learning enthusiast with a passion for building robust, explainable AI systems from the ground up. This project demonstrates my ability to implement complex machine learning algorithms without relying on high-level abstractions.

- **LinkedIn:** [Connect with me](https://www.linkedin.com/in/lokeshkhadke)
- **Email:** [lkhadke32@outlook.com](mailto:lkhadke32@outlook.com)
- **GitHub:** [View more projects](https://github.com/yourusername)

*This project was completed as part of the MSc in Artificial Intelligence at the University of Surrey. I am actively seeking opportunities in cybersecurity analytics, data science, and machine learning engineering where I can apply my skills to real-world challenges.*

---

### 📋 Project Structure

```
network-ids-ml/
├── main.ipynb                 # Complete implementation notebook
├── UNSWNB15_training.csv      # Training data (20,000 samples)
├── UNSWNB15_testing1.csv      Testing data 1 (4,000 samples)
├── UNSWNB15_testing2_no_label.csv # Testing data 2 (25 samples)
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

### 🚀 Getting Started

1. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib
   ```

2. **Run the project:**
   ```bash
   jupyter notebook main.ipynb
   ```

3. **View results:**
   - The notebook contains complete code, explanations, and visualizations
   - Final predictions are shown for all test datasets

---

**Note:** This implementation uses only basic Python libraries as required by the coursework constraints. For production systems, frameworks like TensorFlow or PyTorch would be recommended for maintainability and performance.
