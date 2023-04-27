import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cmath import pi
import warnings
warnings.filterwarnings("ignore")
import pickle
from statsmodels.tsa.arima.model import ARIMA


#pickled_model_1 = pickle.load(open("model_OLS.pkl", 'rb'))

###################################################################################
st.title('Inventory Optimization of Steel Rods | Forecasting')


uploaded_file = st.file_uploader(" ", type=['csv']) #Only accepts csv file format

if uploaded_file is not None:     

    data = pd.read_csv(uploaded_file)
    st.dataframe(data)
######################################################################################
    #fig = plt.figure(figsize = (16,6))
    #sns.lineplot(x = data[data.columns[0]], y = data[data.columns[1]])
    #st.pyplot(fig)

    sd = st.selectbox(
        "Select a Plot", #Drop Down Menu Name
        [
            "Line Plot", #First option in menu
            "Hist Plot"  #Seconf option in menu
        ]
    )

    fig = plt.figure(figsize=(12, 6))

    if sd == "Line Plot":
        sns.lineplot(x = data[data.columns[0]], y = data[data.columns[1]])
            
    elif sd == "Hist Plot":
        sns.histplot(x = data[data.columns[1]])


    st.pyplot(fig)
#####################################################################################

    data["t"] = np.arange(1,len(data) + 1)
    data["t2"] = data["t"]**2
    data["log_value"] = np.log(data[data.columns[1]])
    Sin = np.sin((2 * pi * data["t"]) / 52.143)
    Cos = np.cos((2 * pi * data["t"]) / 52.143)

    data["Sin"] = Sin
    data["Cos"] = Cos

    #st.dataframe(data)

    pickled_model_1 = pickle.load(open("model_OLS.pkl", 'rb'))
    pred_OLS = pickled_model_1.predict(data)
    #st.dataframe(pred_OLS)
#######################################################################################

    prediction_df = data.iloc[:,[0]]
    prediction_df["t"] = np.arange(1,len(data) + 1)
    prediction_df["t2"] = prediction_df["t"]**2
    
    
    Sin = np.sin((2 * pi * prediction_df["t"]) / 52.143)
    Cos = np.cos((2 * pi * prediction_df["t"]) / 52.143)

    prediction_df["Sin"] = Sin
    prediction_df["Cos"] = Cos
    
    pred_on_predict_data_dataset = pickled_model_1.predict(prediction_df)

    residual = data[data.columns[1]] - pickled_model_1.predict(data)
#####################################################################################
    model = ARIMA(residual, order=(1,1,1))
    results = model.fit()
    forecast = results.predict(start = len(residual), end = len(residual) + len(prediction_df) - 1, typ='levels').rename('Forecast')
    forecast.reset_index(drop = True, inplace = True)

    final_df = pred_on_predict_data_dataset + forecast

    df = pd.DataFrame(columns = ["week", "prediction"])
    df["week"] = data.iloc[:,[0]]
    df["prediction"] = np.round(final_df, decimals = 2)
    #df
    st.title('Prediction Values for next 52 weeks: ')
    st.dataframe(df)

    fig = plt.figure(figsize = (16,6))
    sns.lineplot(x = df[df.columns[0]], y = df[df.columns[1]])
    st.pyplot(fig)

########################################################################

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df)

    st.download_button("Press to Download",
                       csv,
                       "file.csv",
                       "text/csv",
                       key = 'download-csv')
