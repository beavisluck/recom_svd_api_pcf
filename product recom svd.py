#there's a possibility of an unused lib here
import numpy as np
import pandas as pd

import random
from sklearn.preprocessing import MinMaxScaler
import os
from flask_restful import Resource,Api
from flask import Flask,request,jsonify
import cx_Oracle
from sqlalchemy import create_engine
from datetime import timedelta
import math
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
from numpy import load
from numpy import save
from flask_cors import CORS
import requests
import json

random.seed(0)
np.random.seed(0)


app=Flask(__name__)
api=Api(app)
here = os.path.dirname(os.path.abspath(__file__))
cors = CORS(app, resources={r"/*": {"origins": "*"}})
cf_port = os.getenv("PORT")
cx_Oracle.init_oracle_client(lib_dir=r"/home/vcap/app/oracle/instantclient")
dsn_tns = cx_Oracle.makedsn('fill ur conn properties here', 'fill ur conn properties here', service_name='fill ur conn properties here') 


cstr = 'fill ur conn string here,just google it'.format(
            sid=dsn_tns
        )

#this is for training the model, always run this first before predict, this will be automated with scheduler
class RecoTrain(Resource):
    def get(self):
        # "with" used to turn off connection manually
        with cx_Oracle.connect(user='fill urr conn properties here', password='fill urr conn properties here', dsn=dsn_tns) as conn:
            curr=conn.cursor()            
            query='select * from company_transaction'
            #load product trans
            df_prod_trans = pd.read_sql(query,con=conn)
            df_prod_trans=df_prod_trans[["COMPANY_ID","PRODUCT_ID"]]
            
            #make a company rating data
            df_comp_rating=df_prod_trans.groupby(["COMPANY_ID","PRODUCT_ID"]).size().reset_index(name='counts')
            del df_prod_trans
            scaler=MinMaxScaler()
            scaler.fit(df_comp_rating["counts"].values.reshape(-1,1))
            df_comp_rating["counts"]=scaler.transform(df_comp_rating["counts"].values.reshape(-1,1))
            quantile=df_comp_rating["counts"].quantile([0.2,0.4,0.6,0.8])
            df_comp_rating["RATINGS"]=np.where(df_comp_rating["counts"]<=quantile.iloc[3],4,5)
            df_comp_rating["RATINGS"]=np.where(df_comp_rating["counts"]<=quantile.iloc[2],3,df_comp_rating["RATINGS"])
            df_comp_rating["RATINGS"]=np.where(df_comp_rating["counts"]<=quantile.iloc[1],2,df_comp_rating["RATINGS"])
            df_comp_rating["RATINGS"]=np.where(df_comp_rating["counts"]<=quantile.iloc[0],1,df_comp_rating["RATINGS"])
            del df_comp_rating["counts"]
            #insert into db for predict
          
            c_alchemy = create_engine(cstr)
            df_comp_rating.to_sql('comp_rating', c_alchemy, if_exists='replace',index=False)
            
            #pivot for training
            df_pivot=df_comp_rating.pivot_table(df_comp_rating, index="COMPANY_ID",columns="PRODUCT_ID",)
            df_pivot=df_pivot.fillna(0)
            R = df_pivot.to_numpy()
            
            #start to produce matrix
            user_ratings_mean = np.mean(R, axis = 1)
            R_demeaned = R - user_ratings_mean.reshape(-1, 1)
            U, sigma, Vt = svds(R_demeaned, k = 8)
            sigma = np.diag(sigma)
            
            #predicted rating matrix
            all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
            sdir=here.replace("\\","//")
            save(sdir+'/reco.npy', all_user_predicted_ratings)
            
            return ("done")

#use to predict the rating 
class RecoPred(Resource):
    def get(self,m_id):
        try:
            #jwt checking
            token=request.headers["Authorization"]
            python_checker="http://zuul-proxy.apps.pcf.dti.co.id/gateway/login-service/v1/auth/pythonChecker"
            
            r = requests.get(python_checker,headers={'Authorization': 'Bearer '+ token[7:]})
            res=r.json()
            if res["success"]==True:
                #to closse conn automaticly
                with cx_Oracle.connect(user='fill urr conn properties here', password='fill urr conn properties here', dsn=dsn_tns) as conn:
                    curr=conn.cursor()
                    comp_id=int(m_id)
                    query='select distinct(company_id) from comp_rating'
                    #load company rating
                    df_company = pd.read_sql(query,con=conn).values
                    # if the company never use product before
                    if (comp_id in df_company) == False:
                        query='select * from company_transaction'
                        #load product trans
                        df_prod_trans = pd.read_sql(query,con=conn)
                        df_prod_trans=df_prod_trans[["COMPANY_ID","PRODUCT_ID"]]
                        df_prod_trans=df_prod_trans.groupby("PRODUCT_ID").size().reset_index(name='counts')
                        df_prod_trans=df_prod_trans.sort_values('counts', ascending=False)
                        final_recomendation=df_prod_trans["PRODUCT_ID"].head(3)
                        #recommend most used product
                        recommended_df=tuple(final_recomendation)
                    
                    else:
                        #load matrix
                        sdir=here.replace("\\","//")
                        all_user_predicted_ratings=load(sdir+"/reco.npy")
                        user_ratings=all_user_predicted_ratings[comp_id-1]
                        
                        #load rating
                        query='select * from comp_rating'
                        df_comp_rating = pd.read_sql(query,con=conn)
                        
                        
                        #load product
                        query='select * from product_dim'
                        df_product_dim = pd.read_sql(query,con=conn)
                        
                        # get the product which the company used
                        prod_used=df_comp_rating.loc[df_comp_rating["COMPANY_ID"]==comp_id,"PRODUCT_ID"].values
                        #prod_used=prod_used["PRODUCT_ID"].values
                        df_product_dim["RATINGS"]=user_ratings
                        del df_product_dim["PRODUCT_NAME"]
                        # get the rating of the used product                      
                        recommended_df=df_product_dim[~df_product_dim["PRODUCT_ID"].isin(prod_used)].sort_values(by="RATINGS",ascending=False)
                        #recommend top 3 product
                        recommended_df=recommended_df["PRODUCT_ID"].head(3)
                        recommended_df=tuple(recommended_df)
                    
                    return jsonify(recommended_df)
            else:
                #if invalid key
                return(["invalid token"])
            
            
        #if key error
        except KeyError :
            return (["invalid token"])
        


        
            
api.add_resource(RecoTrain,'/recotrain')
api.add_resource(RecoPred,'/recopred/<m_id>')        

if __name__ == '__main__':
	if cf_port is None:
		app.run( host='0.0.0.0', port=5000, debug=True, threaded=True )
	else:
		app.run( host='0.0.0.0', port=int(cf_port), debug=True, threaded=True)