import streamlit as st

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import shap
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Forest Fire Area Detection app')
st.write('---')


data = pd.read_csv('forestfires.csv')
if st.checkbox('Show Raw Data'):
    st.subheader('Showing Raw Data-->')
    st.write(data.head())

st.write('Shape of data:')   

target = 'area'

data_new = data.drop(columns=target)

#separating data into numerical and categorical columns
cate_data = data_new.select_dtypes(include='object').columns.tolist()
num_data = data_new.select_dtypes(exclude='object').columns.tolist()

out_col = ['area','FFMC','ISI','rain']
np.log1p(data[out_col]).skew()
np.log1p(data[out_col]).kurtosis()

remov = data.loc[:,['FFMC']].apply(zscore).abs() < 3
data = data[remov.values]
data.shape

#print(data['rain'].value_counts())

#Since almost all the values in 'rain' features is zero.so we convert the rain column into categorical column
#data['rain'] = data['rain'].apply(lambda x: int(x > 0.0))

#out_col.remove('rain')
data[out_col] = np.log1p(data[out_col])

data[out_col].skew()

data[out_col].kurtosis()

dummy_data = data.copy()

#label encoding the categorical column


x = dummy_data.drop(columns = 'area')
y = dummy_data['area']

st.sidebar.header('Specify Input Parameters')
st.sidebar.write('---')

le = LabelEncoder()

def user_input_features():

    st.sidebar.write('X-axis spatial coordinate within the Montesinho park map: 1 to 9')
    X = st.sidebar.selectbox('X',(1,2,3,4,5,6,7,8,9))
    st.sidebar.write('Y-axis spatial coordinate within the Montesinho park map: 1 to 9')
    Y = st.sidebar.selectbox('Y',(1,2,3,4,5,6,7,8,9))
    #X = st.sidebar.slider('X',X.min(),X.max(),1)
    #Y = st.sidebar.slider('Y', list(range(1,10)))
    st.sidebar.write('Month of the year: January to December')
    month = st.sidebar.selectbox('Month',('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'))
    if(month=='jan'):
        month=2
    elif(month=='feb'):
        month=8
    elif(month=='mar'):
        month=10
    elif(month=='apr'):
        month=5
    elif(month=='may'):
        month=3
    elif(month=='jun'):
        month=7
    elif(month=='jul'):
        month=9   
    elif(month=='aug'):
        month=12
    elif(month=='sep'):
        month=11
    elif(month=='oct'):
        month=6  
    elif(month=='nov'):
        month=1    
    else:
        month=4

    
    
    st.sidebar.write('Day of the week: Sunday to Saturday')
    day = st.sidebar.selectbox('Day',('sun','mon','tue','wed','thu','fri','sat'))
    if(day=='sun'):
        day=7
    elif(day=='fri'):
        day=6   
    elif(day=='sat'):
        day=5
    elif(day=='mon'):
        day=4
    elif(day=='tue'):
        day=3  
    elif(day=='thu'):
        day=2    
    else:
        day=1
        
    
    st.sidebar.write('FFMC index from the FWI system:')
    FFMC = st.sidebar.slider('FFMC', 0.0, 4.58, 0.01)
        
    st.sidebar.write('DMC index from the FWI system:')
    DMC = st.sidebar.slider('DMC', 0.00, 291.30,0.1)
       
    st.sidebar.write('DC index from the FWI system:')
    DC =st.sidebar.slider('DC', 0.0, 860.60, 2.00)
       
    st.sidebar.write('ISI index from the FWI system:')
    ISI = st.sidebar.slider('ISI', 0.0, 4.04, 0.01)
       
    st.sidebar.write('Temperature in Celsius degrees:')
    temp =st.sidebar.slider('Temperature',0.0, 33.30, 0.11)
       
    st.sidebar.write('Relative humidity in %:')
    RH = st.sidebar.slider('RH', 1, 99,1)
       
    st.sidebar.write('Wind speed in km/h:')
    wind = st.sidebar.slider('Wind', 0.00,9.40,0.05)
       
    st.sidebar.write('Outside Rain in mm/m2:')
    rain = st.sidebar.slider('Rain', 0.00, 1.0,0.1)

    dat = {'X': X,
            'Y': Y,
            'month': month,
            'day': day,
            'FFMC': FFMC,
            'DMC': DMC,
            'DC': DC,
            'ISI': ISI,
            'temp': temp,
            'RH': RH,
            'wind': wind,
            'rain': rain,}
    features = pd.DataFrame(dat, index=[0])
    
    return features

df=user_input_features()



le = LabelEncoder()
x['day'] = le.fit_transform(x['day'])
x['month'] = le.fit_transform(x['month'])
st.write('---')
st.subheader('User specified Input parameters')
st.write(df)
st.write('---')

df['rain'] = df['rain'].apply(lambda x: int(x > 0.0))




X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=3)

xgb_model = xgb.XGBRegressor(base_score=0.3, booster='gbtree', colsample_bylevel=1,colsample_bytree=0.24, gamma=0,
                             learning_rate=0.01, max_delta_step=0,
                             max_depth=3, min_child_weight=1, missing=None, n_estimators=102,
                             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
                             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,  subsample=1)

eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_metric=["rmse"],eval_set=eval_set, verbose=False)
pred = xgb_model.predict(X_test)
predictions=xgb_model.predict(df)

def calc_ISE(X_train, y_train, model):
    '''returns the in-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return model.score(X_train, y_train), rmse
    
def calc_OSE(X_test, y_test, model):
    '''returns the out-of-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return model.score(X_test, y_test), rmse

is_r2, ise = calc_ISE(X_train, y_train,xgb_model )
os_r2, ose = calc_OSE(X_test, y_test, xgb_model)

# show dataset sizes
data_list = (('R^2_in', is_r2), ('R^2_out', os_r2), 
             ('ISE', ise), ('OSE', ose))
for item in data_list:
    print('{:10}: {}'.format(item[0], item[1]))

rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE: %f" % (rmse))
st.header('Model Training,Evaluation,Prediction')
st.write('Model used: XGBRegressor')
st.write('Metrics used for predictions: RMSE')
st.write('RMSE on raw data is:',rmse)



if st.checkbox('Feature Importance'):
    plt.title('Feature importance based on SHAP values')
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values,X_train,plot_type='bar')
    st.pyplot(bbox_inches='tight',width=5)


# retrieve performance metrics
if st.checkbox('Performance of Training and Test set of the given data'):
    results = xgb_model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    # plot RMSE
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()
    plt.ylabel('RMSE')
    plt.title('XGBoost RMSE')
    st.pyplot()

st.write('---')


st.subheader('Predictions on user specified input parameters ')
st.write('The burned area of the forest (in ha):')
st.write(predictions)
st.write('---')
