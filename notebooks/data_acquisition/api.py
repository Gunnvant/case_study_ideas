from flask import Flask, request,jsonify
import pandas as pd 

russia = pd.read_csv("../../data/data_acquisition/russian_losses_aggregates.csv")
ukraine = pd.read_csv("../../data/data_acquisition/ukrainian_losses_aggregates.csv")

app = Flask(__name__)

@app.route("/russia",methods=['POST'])
def get_data():
   request_data = request.get_json()
   if request_data.get('category') is None:
       return jsonify({'status':400,'message':'category field expected in payload'})
   elif request_data.get('category') not in russia['Category'].unique():
       return jsonify(
           {
               'status':400,
               'message':f"invalid category, category must be one of {russia['Category'].unique()}"
           }
        )
   else:
       cat = request_data.get('category')
       resp = russia[russia['Category']==cat]
       r = {
           'category':cat,
           'details':
            {
               'total':float(resp['Total'].values[0]),
               'destroyed':float(resp['Destroyed'].values[0]),
               'damaged':float(resp['Damaged'].values[0]),
               'abandoned':float(resp['Abandoned'].values[0]),
               'captured':float(resp['Captured'].values[0])
            }
        }
       print(r)
       return jsonify(r)

if __name__=="__main__":
    app.run()
