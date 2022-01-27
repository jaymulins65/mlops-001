# Import libraries
import os
import logging
from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

PATH = os.getcwd()

logging.basicConfig(filename=PATH + '/logs/predict_logs.log', 
                    format='%(asctime)s::%(levelname)s::%(message)s::', 
                    level=logging.INFO)

try:
	clf = load(PATH + '/pickle/model.pkl')
except Exception as e:
        logging.error(e)        
        raise e
else:
        logging.info('Loaded modle.pkl file')

try:
	data = pd.read_csv(PATH + "/data/input/test_cases.csv")
except Exception as e:
        logging.error(e)        
        raise e
else:
        logging.info('Read test_cases.csv file')

try:
	y = data['y']

	data.drop('y', axis=1, inplace=True)
except Exception as e:
        logging.error(e)        
else:
        y_present = True
        logging.info('Droped y attributed')

try:
        y_pred = clf.predict(data)

        if(y_present):
                data['y'] = y
        
        data["y_pred"] = y_pred

except Exception as e:
        logging.error(e)        
else:
        logging.info('Predicted y_hat')

        if(y_present):
                logging.info("Accuracy = %f", accuracy_score(y, y_pred))
                logging.info("Recall   = %f", recall_score(y, y_pred, pos_label='yes'))

        logging.info('Writing test_predict.csv file')
        data.to_csv(PATH + '/data/output/test_predict.csv', index=False, header=True, encoding='utf-8')