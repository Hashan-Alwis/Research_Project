from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

model = load_model('lstm_model/model_checkpoint.keras')



temp_mean  =26.9833575
temp_std =1.1989638869431183

app_temp_mean  =31.186949375
app_temp_std =1.7237414593514624

winspeed_mean =17.559555625
winspeed_std =7.03999711044194

windgusts_mean =35.061196875
windgusts_std =10.01076992568455

radiation_mean =18.866151937500003
radiation_std =4.256934336825035

precipitation_mean =5.8803775
precipitation_std =9.765971710869009

winddirection_mean =189.51228125
winddirection_std =92.75392107167706

eva_mean =4.05091025
eva_std =1.0826466599010673

longitude_mean =80.17879742020001
longitude_std =0.42095185937602364

elevation_mean =10.0721625
elevation_std =5.082428560599131


def preprocess(X):
  X[:, :, 0] = (X[:, :, 0] - temp_mean) /temp_std
  X[:, :, 1] = (X[:, :, 1] - app_temp_mean) / app_temp_std
  X[:, :, 2] = (X[:, :, 2] - winspeed_mean) / winspeed_std
  X[:, :, 3] = (X[:, :, 3] - windgusts_mean) / windgusts_std
  X[:, :, 4] = (X[:, :, 4] - radiation_mean) / radiation_std
  X[:, :, 5] = (X[:, :, 5] - precipitation_mean) / precipitation_std
  X[:, :, 6] = (X[:, :, 6] - winddirection_mean) / winddirection_std
  X[:, :, 7] = (X[:, :, 7] - eva_mean) / eva_std
  X[:, :, 8] = (X[:, :, 8] - longitude_mean) / longitude_std
  X[:, :, 9] = (X[:, :, 9] - elevation_mean) / elevation_std


def reverse_preprocessed_temp(arr):
    arr = (arr * temp_std ) + temp_mean
    return arr

def reverse_preprocessed_app_temp(arr):
    arr = (arr * app_temp_std ) + app_temp_mean
    return arr

def reverse_preprocessed_winspeed(arr):
    arr = (arr * winspeed_std ) + winspeed_mean
    return arr

def reverse_preprocessed_windgust(arr):
    arr = (arr * windgusts_std ) + windgusts_mean
    return arr




def get_next_day_whether(df, model=model):

    if len(df)!=4:
        print("data should contain 4 rawos")
        return()

    data=df
    data['Seconds'] = data.index.map(pd.Timestamp.timestamp)

    day = 60*60*24
    year = 365.2425*day

    data['Day sin'] = np.sin(data['Seconds'] * (2* np.pi / day))
    data['Day cos'] = np.cos(data['Seconds'] * (2 * np.pi / day))
    data['Year sin'] = np.sin(data['Seconds'] * (2 * np.pi / year))
    data['Year cos'] = np.cos(data['Seconds'] * (2 * np.pi / year))

    data.drop("Seconds", axis=1, inplace=True)


    def convert_df_input_arr(df, window_size=4):
        df_as_np = df.to_numpy()
        x = []
        row = [r for r in df_as_np[0:window_size]]
        # print(row)
        x.append(row)

        return np.array(x)
    
    p= convert_df_input_arr(data)
   
    preprocess(p)

    scaled_prediction= model.predict(p)

    temp_predict = reverse_preprocessed_temp(scaled_prediction[:, 0])
    app_temp_predict = reverse_preprocessed_app_temp(scaled_prediction[:, 1])
    winspeed_predict = reverse_preprocessed_winspeed(scaled_prediction[:, 2])
    windgust_predict = reverse_preprocessed_windgust(scaled_prediction[:, 3])
    
    # print(
    #     f"Temperature Prediction: {temp_predict[0]:.1f}\n"
    #     f"Apparent Temperature Prediction: {app_temp_predict[0]:.1f}\n"
    #     f"Wind Speed Prediction: {winspeed_predict[0]:.1f}\n"
    #     f"Wind Gust Prediction: {windgust_predict[0]:.1f}"
    # )

    
    return(temp_predict, app_temp_predict, winspeed_predict, windgust_predict)