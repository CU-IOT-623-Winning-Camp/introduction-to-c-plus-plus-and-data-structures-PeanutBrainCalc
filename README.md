# Fraud Detection System

This project detects fraudulent transactions using Neural Networks. Additionally it displays the usage of Random Forest for Fraud Detection.
![HeatMapRandomForest]()

## Datasets Used
**[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** 
**[Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/avikpaul4u/vehicle-loan-default-prediction)**
**[Fraud ecommerce](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce)**

## Initial Setup
- - Clone the repository and install the dependencies using `pip install -r requirements.txt`or dowlnoad the zip file in a suitable directory on your local machine and unzip it using the `Expand-Archive -Path 'DownlaodedZipFileName.zip' -DestinationPath 'NewFolderName` command on the powershell/command prompt on windows.

- Navigate to the Folder manually or windows users can use the command prompt with the command: 
-- `cd *folderpath*`
-- if done manually, then for the next step right click inside the folder and select open in terminal

- Ones that folder is open create a virtual environment in the folder path for the project using the commands: 

```
python -m venv venv
venv\Scripts\activate
```
- Install the [necessary libraries]() for this project in the virtual environment using the command:
` pip install pandas scikit-learn matplotlib seaborn tensorflow kaggle imbalanced-learn`

These steps ensure that the environment base is setup for further evaluations of the project using python/ VS Code.
Ones done open the folder in VS Code or use the following command in the terminal:
`code .`

## Troubleshooting

-- If required try replacing the the file paths wherever mentioned in the code to approprite paths according to file location in your remote device. For example: The file "CreditCardRandomForestScript.py" contains the codeline: `df = pd.read_csv('./creditcard.csv')` . So Make sure that the creditcard.csv is either present directly in the project folder or replace './creditcard.csv 'part  with the complete appropriate url of this file specific to your device.



## Steps Involved in Creation of this Project:
- Load the dataset
- Preprocess the data (handling missing values, scaling features)
- Apply SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance
- Train a Random Forest classifier
- Save the model as `fraud_detection_final_model.pkl`
- Apply Neural Networks on each of the datasets.
- Evaluate the model performance using metrics such as accuracy, precision, recall, and F1-score
- Save the nn model as with a `.h5` extention.


## Libraries Used:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imblearn`
- `joblib`

## How to Run:

- Run `python CreditCardRandomForestScript.py` to train the model and make predictions.
- Run `python ecommercefraudScript.py` to train the model and make predictions.
- Run `python creditcardScript.py` to train the model and make predictions.
- Run `python vehicle-loanScript.py` to train the model and make predictions.




[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/BzSVdrny)
