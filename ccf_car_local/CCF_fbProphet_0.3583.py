# Author : LiangQuanSheng
#  Time  : 2019/8/23 12:05

import pandas as pd
from fbprophet import Prophet
import warnings
from datetime import date
import calendar
import os
import logging
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')

class ProphetPredit:
    def __init__(self,Train,Submit):
        self.Train = Train
        self.Submit  = Submit
        self.Holiday = pd.DataFrame({
            "holiday":["NewYear","Spring","Labour","National","NewYear","Spring"],
            "ds":[date(2017,1,31),date(2017,2,28),date(2017,5,31),date(2017,10,30),date(2018,1,31),date(2018,2,28)],
            "lower_window":-1,
            "upper_window":1
        })

    def GetData(self,adcode,model):
        data = self.Train[(self.Train["adcode"] == adcode) & (self.Train["model"] == model)]
        data ["ds"] = data[["regYear", "regMonth"]].apply(lambda x: date(x["regYear"], x["regMonth"], calendar.monthrange(x["regYear"],x["regMonth"])[1]), axis=1)
        data ["y"]  = data["salesVolume"]
        return data

    def GetPredict(self,adcode,model):
        m = Prophet(yearly_seasonality=True,holidays_prior_scale=35,holidays=self.Holiday)
        m.add_country_holidays(country_name="CN")
        data = self.GetData(adcode,model)
        m.fit(data[["ds", "y"]])
        future = m.make_future_dataframe(periods=4, freq="M")
        forecast = m.predict(future)
        score = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4)
        score["score"] = score["yhat_upper"] * 0.6 + score["yhat_lower"] * 0.4
        score["regYear"] = score["ds"].apply(lambda x: x.year)
        score["regMonth"] = score["ds"].apply(lambda x: x.month + 1)
        for month in score["regMonth"].unique():
            self.Submit.loc[(self.Submit["adcode"]==adcode) & (self.Submit["model"]==model) & (self.Submit["regMonth"]==month),"forecastVolum"] = round(score["score"][score["regMonth"]==month].values[0],0)
        print(f"{adcode} {model} Predict Done Result:{round(score['score'][score['regMonth']==month].values[0],0)}......")

    def main(self):
        adcode = self.Train["adcode"].unique()
        model = self.Train["model"].unique()
        for Adcode in adcode:
            for Model in model:
                self.GetPredict(Adcode,Model)
        return self.Submit

if __name__ == '__main__':
    os.chdir("./ccf_car/")
    Sale = pd.read_csv("train_sales_data.csv")
    Search = pd.read_csv("train_search_data.csv")
    UserReply = pd.read_csv("train_user_reply_data.csv")
    Test = pd.read_csv("evaluation_public.csv")
    data = Sale.merge(Search, how="left", on=["province", "adcode", "model", "regYear", "regMonth"])
    data = data.merge(UserReply, how="left", on=["model", "regYear", "regMonth"])
    P=ProphetPredit(data,Test)
    Submain=P.main()
    Submain.to_csv("/mnt/file/competition/ccf_car/rst/prophet_rst.csv",encoding='gbk',index=False)











