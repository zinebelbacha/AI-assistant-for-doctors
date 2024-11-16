# AI Assistant for Doctors

This repository contains two advanced AI models designed to assist doctors in diagnosing and analyzing medical conditions from chest X-ray images and medical reports. Below are the details of the two models:

---

## **1. Chest X-ray Classification Model**

This model is a custom TensorFlow implementation for classifying chest X-ray images, leveraging the DenseNet121 architecture.

### **Model Details**
- **Architecture**: DenseNet121
- **Pre-trained on**: CheXpert dataset
- **Input size**: 224x224 pixels
- **Number of classes**: 14 (common chest X-ray findings)

### **Compatible Datasets**
1. **NIH Chest X-ray Dataset**  
   - **Source**: [Kaggle](https://www.kaggle.com/nih-chest-xrays/data)  
   - **Contents**: 112,120 X-ray images from 30,805 unique patients  

2. **MIMIC-CXR Database**  
   - **Source**: [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/)  
   - **Contents**: 377,110 chest X-rays associated with 227,835 imaging studies  

---

## **2. OCR Model for Medical Report Analysis**

This model is designed to extract and analyze textual information from medical reports. It classifies the type of case based on the extracted information, identifying conditions such as tuberculosis, pneumonia, or normal cases.

### **Key Features**
- **OCR Capability**: Reads medical analyses and identifies relevant clinical findings.
- **Supported Conditions**:  
  - Tuberculosis  
  - Pneumonia  
  - Normal  

---

### **Usage**

To use these models effectively, follow the instructions in the documentation provided in this repository.

1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Preprocess the datasets using the scripts provided.
4. Run the models for classification or text analysis.

---

For further details, refer to the detailed documentation in each model's folder.
