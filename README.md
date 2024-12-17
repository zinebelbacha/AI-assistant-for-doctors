![image](https://github.com/user-attachments/assets/0dbfcceb-e74c-415b-99a2-5c80645ff06b)


# What is ai assistant for doctors

AI assistant for doctors contains two advanced AI models designed to assist doctors in diagnosing and analyzing medical conditions from chest X-ray images and medical reports. Below are the details of the two models:
![image](https://github.com/user-attachments/assets/f1bedf5d-8bf2-456b-b64d-64c303490806)



---

## **1. Chest X-ray Classification Model**

This model is a custom TensorFlow implementation for classifying chest X-ray images, leveraging the DenseNet121 architecture , fine tuned on the dataset below .

### **Model Details**
- **Architecture**: DenseNet121
- **Pre-trained on**: CheXpert dataset
- **Input size**: 224x224 pixels
- **Number of classes**: 14 (common chest X-ray findings)

### **Compatible Datasets**
1. **NIH Chest X-ray Dataset**  
   - **Source**: [Kaggle](https://www.kaggle.com/nih-chest-xrays/data)  
   - **Contents**: 112,120 X-ray images from 30,805 unique patients  

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


## **3. Models Process Overview
### ** X-Ray Image Preprocessing 
Preprocess the datasets using the scripts provided.
Run the models for classification or text analysis.

---

For further details, refer to the detailed documentation in each model's folder.
### ** Medical Reports Preprocessing 
 - Suppression des caractères non pertinents
 - Gestion des formats
 - Anonymisation
 - Filtrage des stopwords
 - Extraction d'entités médicales
## **4.Deep Face Model for authentification
To enhance the security of the website, we have implemented two-factor authentication to ensure that only the authorized doctor can access the sensitive medical dataset. Below is an overview of the website and an image stored by our Deep Face Model as part of the authentication process.
![image](https://github.com/user-attachments/assets/521ebffb-a2e2-47dd-b950-b877205256e0)
![image](https://github.com/user-attachments/assets/a69ebd95-dab6-4d13-b85e-cbede92e7239)
