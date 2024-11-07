from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib 
import pandas as pd

model = joblib.load('loan_status_predictor.pkl')

app = FastAPI()

num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = joblib.load('vector.pkl')

# Gender	Married	Dependents	Education	Self_Employed	ApplicantIncome	CoapplicantIncome	LoanAmount	Loan_Amount_Term	Credit_History	Property_Area

class LoanApproval(BaseModel):
    Gender: float 
    Married: float
    Dependents:float
    Education: float
    Self_Employed: float
    ApplicantIncome:float
    CoapplicantIncome:float
    LoanAmount:float 
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: float

@app.post("/predict")
async def predict_loan_status(application: LoanApproval):
    input_data = pd.DataFrame([application.dict()])
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    result = model.predict(input_data)

    if result[0] == 1:
        return {'Loan Status': "Approved"}
    else:
        return {'Loan Status': "Not Approved"}