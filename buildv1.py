import os
import logging
import configparser
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from joblib import dump
# from sklearn.metrics import accuracy_score, recall_score
# import datetime

PATH = os.getcwd()

logging.basicConfig(filename=PATH + '/logs/build_logs.log', 
                    format='%(asctime)s::%(levelname)s::%(message)s::', 
                    level=logging.INFO)

try:
	config = configparser.ConfigParser()
	config.read(PATH+'/conf/config.ini')

	host        = config['MySQL']['host']
	port        = config['MySQL']['port']
	user        = config['MySQL']['user']
	password    = config['MySQL']['password']
	db          = config['MySQL']['db']

	svm_c       = int(config['Model']['c'])
	svm_kernel  = config['Model']['kernel']
	svm_gamma   = float(config['Model']['gamma'])

	cat_Attr_Names = config['Dtypes']['category'].split(',')
	num_Attr_Names = config['Dtypes']['float64'].split(',')
except Exception as e:
        logging.error(e)        
        raise e
else:
        logging.info('Read config.ini file')
	
		
try:
	connector = 'mysql+mysqlconnector://' + str(user) + ':' + str(password) + '@' + str(host) + ':' + str(port) + '/' + str(db)

	data = pd.read_sql("select * from bank", con=connector)
except Exception as e:
        logging.error(e)        
        raise e
else:
        logging.info('Read data from mysql')


try:
	data.replace(to_replace=['unknown'], value=np.nan, inplace=True)

	# customer_no is not of much value to dropping it
	data = data.drop(['customer_no'], axis=1)

	# Convert attributes into appropriate type
	data[cat_Attr_Names] = data[cat_Attr_Names].apply(lambda col: col.astype('category'))
	data[num_Attr_Names] = data[num_Attr_Names].apply(lambda col: col.astype('float64'))

	X = data.drop('y', axis=1)
	y = np.array(data['y'])

	cat_Attr_Names.remove('y')
except Exception as e:
        logging.error(e)        
        raise e
else:
        logging.info('Data Prepared for Training')

try:
	numeric_transformer = Pipeline(steps=[
	    ('imputer', SimpleImputer(strategy='mean')),
	    ('scaler', StandardScaler())])

	categorical_transformer = Pipeline(steps=[
	    ('imputer', SimpleImputer(strategy='most_frequent')),
	    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

	preprocessor = ColumnTransformer(
	    transformers=[
		('num', numeric_transformer, num_Attr_Names),
		('cat', categorical_transformer, cat_Attr_Names)])

	classifier = SVC(C=svm_c, kernel=svm_kernel, gamma=svm_gamma, class_weight='balanced')

	clf = Pipeline(steps=[
	    ('preprocessor', preprocessor), 
	    ('classifier', classifier)])

	clf.fit(X, y)
except Exception as e:
        logging.error(e)        
        raise e
else:
        logging.info('Data pre-processed and model Trained')

try:
	dump_file = PATH + '/pickle/model.pkl'
	dump(clf, dump_file, compress=1)
except Exception as e:
        logging.error(e)        
        raise e
else:
        logging.info('Saved %s pipeline to %s file' % (clf, dump_file))

# y_pred = clf.predict(X)

# print("Accuracy = ", accuracy_score(y, y_pred))
# print("Recall   = ", recall_score(y, y_pred, pos_label='yes'))

# dump_file = PATH + '/pickle/model'+ datetime.datetime.now().isoformat()+'.pkl'

