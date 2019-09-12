if __name__ == '__main__':
    import sys

if len(sys.argv) == 1:
    print("Please provide the filename or the path to the dataset.")
    filename = input("Enter filename or filepath: ")
else:
    filename = sys.argv[1]

import pickle
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

##### FUNCTIONS - DATA TRANSFORMATION

### Function to transform categorial variables in dummies

def CategoricalToDummies(pd_data, variables):
  dataset = pd_data.copy()
  for variable in variables:
    dummies = pd.get_dummies(dataset[variable]).rename(columns=lambda x: variable + '_' + str(int(round(x,0))))
    #dummies = dummies.drop(dummies.columns[[0]], axis=1) #remove one category. Not recommended for variables with NaN
    dataset = pd.concat([dataset, dummies], axis=1)
    dataset = dataset.drop(variable, axis=1)
  return dataset

### Function to reshape data after scaling (only one column)

def InverseScaler(ttype, data_vector, params):
  #reshaped = []
  if ttype == 'minmax':
    xmin = params[0]
    xmax = params[1]
    reshaped = xmin + data_vector*(xmax-xmin)
  if ttype == 'meanscale':
    xmin = params[0]
    xmax = params[1]
    xmean = params[2]
    reshaped = xmean + data_vector*(xmax-xmin)
  return reshaped

##### FUNCTIONS - MODELING

### Function to predict

def PredictValues(model, X_test, scale = None):
  Y_pred = model.predict(X_test)
  if scale == None:
    Y_pred = [round(value) for value in Y_pred]
  else:
    ttype = scale[0]
    params = scale[1]
    Y_pred = InverseScaler(ttype, Y_pred, params)
    Y_pred = [round(value) for value in Y_pred]
  return Y_pred

# load the model from disk
modelfile = 'model.sav'
model = pickle.load(open(modelfile, 'rb'))
print("Model file loaded.")

# load the dataset from disk. Must be renamed to 'dataset.csv'
dataset = pd.read_csv(filename,sep=',')
print("Data file loaded.")

#select initial variables
print("Transforming data...")
useless = ['x001','x067','x094','x095','x096']
selected = list(set(dataset.columns)-set(useless))
selected = ['x002','x003','x004','x005','x006','x007','x008','x009','x010','x011','x012','x013','x014','x015','x016','x017','x018','x019','x020','x021','x022','x023','x024','x025','x026','x027','x028','x029','x030','x031','x032','x033','x034','x035','x036','x038','x040','x041','x042','x043','x044','x045','x046','x047','x051','x052','x053','x054','x055','x056','x057','x058','x059','x062','x063','x064','x065','x066','x071','x072','x073','x074','x075','x076','x079','x080','x081','x082','x086','x087','x088','x089','x091','x092','x097','x098','x099','x100','x102','x103','x104','x105','x106','x107','x111','x112','x114','x115','x116','x117','x118','x119','x120','x121','x124','x126','x127','x129','x130','x131','x133','x134','x135','x136','x138','x139','x140','x141','x142','x143','x144','x145','x146','x147','x148','x149','x150','x151','x152','x153','x155','x157','x158','x159','x160','x162','x163','x164','x168','x169','x170','x171','x172','x173','x174','x175','x178','x179','x181','x182','x183','x184','x185','x186','x187','x188','x189','x190','x191','x192','x193','x194','x195','x196','x197','x198','x199','x200','x201','x202','x203','x204','x205','x206','x207','x208','x209','x210','x211','x212','x213','x214','x215','x216','x217','x218','x219','x220','x221','x222','x223','x224','x225','x226','x227','x228','x229','x230','x231','x232','x233','x234','x235','x236','x237','x238','x239','x240','x241','x242','x243','x245','x246','x247','x248','x249','x250','x252','x253','x254','x255','x256','x257','x258','x259','x264','x265','x266','x267','x268','x269','x270','x271','x272','x273','x274','x275','x276','x277','x278','x279','x280','x281','x282','x283','x284','x285','x286','x287','x288','x289','x290','x291','x292','x293','x294','x295','x296','x297','x298','x299','x300','x301','x302','x303','x304','y']

selected.sort()
categorical = ['x018','x019','x022','x023','x037','x038','x039','x046','x047','x048','x049','x050','x051','x052','x053','x054','x055','x061','x068','x069','x077','x078','x079','x080','x100','x101','x102','x107','x108','x112','x122','x123','x148','x155','x156','x162','x163','x169','x174','x175','x176','x177','x178','x179','x182','x183','x228','x229','x241','x251','x252','x253','x254','x287','x302']
cat_selected = list(set(categorical).intersection(selected))
cat_selected.sort()

#Transform categorical variables to dummies
DF_test = dataset[selected]
DF_test = CategoricalToDummies(DF_test, cat_selected)
finalvars = ['x002', 'x003', 'x004', 'x005', 'x006', 'x007', 'x008', 'x009', 'x010', 'x011', 'x012', 'x013', 'x014', 'x015', 'x016', 'x017', 'x018_1', 'x018_2', 'x018_3', 'x018_4', 'x019_0', 'x019_1', 'x019_2', 'x019_4', 'x020', 'x021', 'x022_0', 'x022_1', 'x022_2', 'x022_3', 'x023_0', 'x023_1', 'x024', 'x025', 'x026', 'x027', 'x028', 'x029', 'x030', 'x031', 'x032', 'x033', 'x034', 'x035', 'x036', 'x038_1', 'x040', 'x041', 'x042', 'x043', 'x044', 'x045', 'x046_0', 'x046_3', 'x047_3', 'x051_0', 'x051_2', 'x052_0', 'x052_3', 'x053_0', 'x053_2', 'x054_0', 'x055_0', 'x055_1', 'x055_2', 'x055_3', 'x055_4', 'x055_5', 'x055_6', 'x055_7', 'x056', 'x057', 'x058', 'x059', 'x062', 'x063', 'x064', 'x065', 'x066', 'x071', 'x072', 'x073', 'x074', 'x075', 'x076', 'x079_0', 'x080_0', 'x081', 'x082', 'x086', 'x087', 'x088', 'x089', 'x091', 'x092', 'x097', 'x098', 'x099', 'x100_0', 'x100_1', 'x102_0', 'x102_1', 'x103', 'x104', 'x105', 'x106', 'x107_0', 'x111', 'x112_1', 'x114', 'x115', 'x116', 'x117', 'x118', 'x119', 'x120', 'x121', 'x124', 'x126', 'x127', 'x129', 'x130', 'x131', 'x133', 'x134', 'x135', 'x136', 'x138', 'x139', 'x140', 'x141', 'x142', 'x143', 'x144', 'x145', 'x146', 'x147', 'x148_1', 'x148_2', 'x149', 'x150', 'x151', 'x152', 'x153', 'x155_1', 'x157', 'x158', 'x159', 'x160', 'x162_1', 'x162_2', 'x162_5', 'x162_9', 'x163_0', 'x164', 'x168', 'x169_0', 'x169_1', 'x169_6', 'x170', 'x171', 'x172', 'x173', 'x174_0', 'x174_1', 'x175_0', 'x178_0', 'x179_0', 'x179_1', 'x181', 'x182_0', 'x182_1', 'x183_0', 'x183_1', 'x184', 'x185', 'x186', 'x187', 'x188', 'x189', 'x190', 'x191', 'x192', 'x193', 'x194', 'x195', 'x196', 'x197', 'x198', 'x199', 'x200', 'x201', 'x202', 'x203', 'x204', 'x205', 'x206', 'x207', 'x208', 'x209', 'x210', 'x211', 'x212', 'x213', 'x214', 'x215', 'x216', 'x217', 'x218', 'x219', 'x220', 'x221', 'x222', 'x223', 'x224', 'x225', 'x226', 'x227', 'x228_0', 'x228_1', 'x229_0', 'x230', 'x231', 'x232', 'x233', 'x234', 'x235', 'x236', 'x237', 'x238', 'x239', 'x240', 'x241_0', 'x241_1', 'x242', 'x243', 'x245', 'x246', 'x247', 'x248', 'x249', 'x250', 'x252_0', 'x253_1', 'x253_2', 'x254_0', 'x254_1', 'x255', 'x256', 'x257', 'x258', 'x259', 'x264', 'x265', 'x266', 'x267', 'x268', 'x269', 'x270', 'x271', 'x272', 'x273', 'x274', 'x275', 'x276', 'x277', 'x278', 'x279', 'x280', 'x281', 'x282', 'x283', 'x284', 'x285', 'x286', 'x287_1', 'x287_2', 'x287_5', 'x287_7', 'x287_8', 'x287_9', 'x288', 'x289', 'x290', 'x291', 'x292', 'x293', 'x294', 'x295', 'x296', 'x297', 'x298', 'x299', 'x300', 'x301', 'x302_1', 'x302_5', 'x302_9', 'x303', 'x304','y']

DF_test = DF_test[finalvars]
X_test = DF_test.loc[:, DF_test.columns != 'y']
Y_test = DF_test['y']

print("Predicting values...")
# Predict values
Y_pred = PredictValues(model, X_test)
# Calculate RMSE
RMSE = sqrt(mean_squared_error(Y_test, Y_pred))
# Calculate Accuracy
error_flag = 3
GoodOnes = []
PredErrors = abs(Y_pred-Y_test)
for PE in PredErrors:
  if PE <= error_flag:
    GoodOnes.append(1)
  else:
    GoodOnes.append(0)
accuracy = sum(GoodOnes)/len(GoodOnes)

print("RMSE: "+str(round(RMSE,2)))
print("Accuracy: "+str(round(accuracy*100,2))+"%")

print("Saving predictions to file!!")
pd.DataFrame(Y_pred).to_csv("output.csv", header=False, index=False)

input("Press any key to exit: ")