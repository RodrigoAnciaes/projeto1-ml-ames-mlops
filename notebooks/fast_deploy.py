from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import pandas as pd


class AmesMeasurment (BaseModel):
    Mas_Vnr_Area: float         # Em square feet            
    Foundation_Other: bool      # Se a fundação é de outro tipo     
    Fireplaces: int             # de 0 a 4
    Remod_Age: int              # Idade do remodelamento, em anos
    Kitchen_Qual: int           # de 0 a 3
    Garage_Age: int             # Idade da garagem, em anos
    Lot_Area: float             # Em square feet          
    Garage_Cars: int            # de 0 a 4
    X1st_Flr_SF: float          # Em square feet           
    BsmtFin_SF_1: float         # Em square feet            
    Exter_Qual: int             # 0 a 3
    Total_Bsmt_SF: float        # Em square feet            
    Garage_Area: float          # Em square feet               
    Gr_Liv_Area: float          # Em square feet              
    Overall_Qual: float         # de 0 a 7

app = FastAPI()

stack_reg = pickle.load(open('stack_reg.pkl', 'rb'))

@app.get('/')
def index():
    return {'API': 'API build for Insper Machine Learning Course, using FastAPI and Ames Housing Dataset'}

@app.post('/predict')
def get_prediction(data: AmesMeasurment):
    data = data.model_dump()
    mas_vnr_area = data['Mas_Vnr_Area']
    foundation_other = data['Foundation_Other']
    fireplaces = data['Fireplaces']
    remod_age = data['Remod_Age']
    kitchen_qual = data['Kitchen_Qual']
    garage_age = data['Garage_Age']
    lot_area = data['Lot_Area']
    garage_cars = data['Garage_Cars']
    x1st_flr_sf = data['X1st_Flr_SF']
    bsmtfin_sf_1 = data['BsmtFin_SF_1']
    exter_qual = data['Exter_Qual']
    total_bsmt_sf = data['Total_Bsmt_SF']
    garage_area = data['Garage_Area']
    gr_liv_area = data['Gr_Liv_Area']
    overall_qual = data['Overall_Qual']

    sample_data = [[mas_vnr_area, foundation_other, fireplaces, remod_age, kitchen_qual, garage_age, lot_area, garage_cars, x1st_flr_sf, bsmtfin_sf_1, exter_qual, total_bsmt_sf, garage_area, gr_liv_area, overall_qual]]
    prediction = stack_reg.predict(sample_data)

    # Use the values of data to predict the price of the house
    # prediction = stack_reg.predict(data)[0]
    return {
        'preço': prediction[0]
    }

# Index(['Mas_Vnr_Area'_ 'Foundation_Other'_ 'Fireplaces'_ 'Remod_Age'_
#        'Kitchen_Qual'_ 'Garage_Age'_ 'Lot_Area'_ 'Garage_Cars'_ 'X1st_Flr_SF'_
#        'BsmtFin_SF_1'_ 'Exter_Qual'_ 'Total_Bsmt_SF'_ 'Garage_Area'_
#        'Gr_Liv_Area'_ 'Overall_Qual']_
#       dtype='object')