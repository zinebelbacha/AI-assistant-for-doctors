![WhatsApp Image 2025-05-30 at 16 05 01_68bedf56](https://github.com/user-attachments/assets/e9aaba50-1108-47c7-b0a2-9b42cb17e68e)


# What is AI assistant for doctors?

AI assistant for doctors contains two advanced AI models designed to assist doctors in diagnosing and analyzing medical conditions from chest X-ray images and medical reports. Below are the details of the two models:
![image](https://github.com/user-attachments/assets/291496e7-21f7-4c4f-92c1-2f15d2b76960)



---

## **1. Chest X-ray Classification Model**

This model is a custom TensorFlow implementation for classifying chest X-ray images, leveraging the DenseNet121 architecture , fine tuned on a customized dataset.

### **Model Details**
- **Architecture**: DenseNet121
- **Pre-trained on**: CheXpert dataset
- **Input size**: 224x224 pixels
- **Number of classes**: 14 (common chest X-ray findings)



## **2. OCR & Classifier Model for Medical Report Analysis**

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


## **3. Models Process Overview**
### **X-Ray Image Preprocessing**
Preprocess the datasets using the scripts provided.
Run the models for classification or text analysis.


### **Medical Reports Preprocessing**

Here is the translation of the phrases into English:

-Removal of irrelevant characters  

-Format management  

-Anonymization  

-Stopword filtering  

-Extraction of medical entities  

To enhance the security of the website, we have implemented two-factor authentication to ensure that only the authorized doctor can access the sensitive medical dataset. Below is an overview of the website and an image stored by our Deep Face Model as part of the authentication process.


![WhatsApp Image 2025-05-30 at 16 06 51_85826acb](https://github.com/user-attachments/assets/1aa7e6fe-3e24-4e24-b0b0-dbffc565daec)
![WhatsApp Image 2025-05-30 at 16 09 36_98e5234c](https://github.com/user-attachments/assets/ddda21c4-b79f-42b2-bc59-c03ee489527c)
