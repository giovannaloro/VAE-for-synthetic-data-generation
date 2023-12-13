import pandas as pd
import os 
import scipy 
from sklearn import preprocessing
"""
#uplouading dataset 
datasets=['ML_MED_Dataset_train.csv']
for dataset in datasets:
    df = pd.read_csv(f'{dataset}')
    X_num = df.iloc[:,0:4]
    X_cat = df.iloc[:,5:46]
    #scaling and selecting dataset
    scaler_num_train = preprocessing.StandardScaler().fit(X_num)
    X_num = scaler_num_train.transform(X_num)
    X_num = pd.DataFrame(X_num, columns = ['Età','Peso','Altezza','BMI'])
    X = X_num.join(X_cat)
    y = df.iloc[:,47:51]
    output_dataset = X.join(y)
    #saving preprocessed dataset
    file_name = dataset.split('.')[0] + '_preprocessed_scale_complete' + '.csv'
    output_dataset.to_csv(file_name)
"""
def to_label(dataset_name):
    df = pd.read_csv(f'{dataset_name}')
    X = df.iloc[:,1:46]
    y_onehot = df.iloc[:,46:52]
    print(y_onehot.head())
    print(X.head())
    y_label = pd.DataFrame(columns = ['tipo_operazione'])
    for index, row in y_onehot.iterrows():
        operation = (2*row['INTERVENTI SUL SISTEMA ENDOCRINO']+1*row['INTERVENTI SULL’APPARATO DIGERENTE'] +
                    3*row['INTERVENTI SULL’APPARATO URINARIO'] + 4*row['INTERVENTI SUL SISTEMA RESPIRATORIO']+
                    5*row['INTERVENTI SUL SISTEMA CARDIOVASCOLARE'])   
        match operation:
            case 1:
                list_row = [1]
                y_label.loc[len(y_label)] = list_row
            case 2:
                list_row = [2]
                y_label.loc[len(y_label)] = list_row    
            case 3:
                list_row = [3]
                y_label.loc[len(y_label)] = list_row
            case 4:
                list_row = [4]
                y_label.loc[len(y_label)] = list_row
            case 5:
                list_row = [5]
                y_label.loc[len(y_label)] = list_row
    output_dataset=X.join(y_label)
    file_name = dataset_name.split('.')[0] + '_label' + '.csv'
    output_dataset.to_csv(file_name)

files=['ML_MED_Dataset_train_preprocessed_scale_complete.csv']
for file in files:
    to_label(file)
