import pandas as pd 
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split as tts 
import sklearn.ensemble as ensemble 

df = pd.read_csv('penguins.csv')

df.drop(columns = ['island','sex'], inplace = True)
df.rename(columns={'culmen_length_mm': 'culmen_length', 'culmen_depth_mm': 'culmen_depth', 'flipper_length_mm': 'flipper_length'}, inplace=True)
df = df.dropna()

X = df.drop(['species', 'body_mass_g'], axis = 1)
y = df['species']

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = tts(X_smote, y_smote, test_size = 0.3, random_state = 143)

def training_model():
    model = ensemble.ExtraTreesClassifier()
    trained_model = model.fit(X_train, y_train)
    return trained_model