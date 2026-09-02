from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel,Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware


model1 = joblib.load("classifier_pipeline.pkl")
model2 = joblib.load("salary_pipeline.pkl")
le = joblib.load("label_encoder.pkl")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentInput(BaseModel):
    Age: int = Field(..., ge=17, le=35)
    Attendance_Percentage: float = Field(..., ge=0, le=100)
    Study_Hours_Per_Week: float = Field(..., ge=0)
    CGPA: float = Field(..., ge=0, le=4)
    Programming_Skill: int = Field(..., ge=1, le=10)
    Projects_Completed: int = Field(..., ge=0)
    Certifications: int = Field(..., ge=0)
    Hackathons: int = Field(..., ge=0)
    Internships: int = Field(..., ge=0, le=5)
    Communication_Skills: int = Field(..., ge=1, le=10)
    Teamwork: int = Field(..., ge=1, le=10)
    Problem_Solving: int = Field(..., ge=1, le=10)
    Interview_Score: float = Field(..., ge=0, le=100)
    Employability_Score: float = Field(..., ge=0)
    Resume_Score: float = Field(..., ge=0, le=100)

    University_Year: Literal['Freshman', 'Sophomore', 'Junior', 'Senior']
    Academic_Performance: Literal['Poor', 'Average', 'Good', 'Excellent']
    English_Proficiency: Literal['Basic', 'Intermediate', 'Advanced']
    Gender: Literal['Male', 'Female', 'Other']
    Major: Literal['Computer Science', 'Information Technology', 'Software Engineering',
                    'Artificial Intelligence', 'Data Science', 'Cybersecurity',
                    'Business Analytics', 'Electrical Engineering']
    GitHub_Profile: Literal['Yes', 'No']
    Leadership_Experience: Literal['Yes', 'No']
    LinkedIn_Profile: Literal['Yes', 'No']


@app.post("/predict")
def predict(student: StudentInput):
    input_dict = student.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    placement_num = model1.predict(input_df)[0]
    placement_label = str(le.inverse_transform([placement_num])[0])
    
    if placement_label == "Placed":
        salary_prediction = float(model2.predict(input_df)[0])
    else:
        salary_prediction = 0.0  
    
    return {
        "Placement_Prediction": placement_label,
        "Salary_Prediction": salary_prediction
    }