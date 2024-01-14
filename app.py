import streamlit as st 
from src.pipeline.predict_pipeline import CustomData
from src.utils import load_obj
import pandas as pd 
import os

@st.cache_resource
def load_models():
    estimator = load_obj(file_path=os.path.join("artifact", "model.pkl"))
    preprocessor = load_obj(file_path=os.path.join("artifact", "preprocessor.pkl"))
    return estimator, preprocessor

def predict_value(features:pd.DataFrame) -> float:
    estimator, preprocessor = load_models()
    transfomed_features = preprocessor.transform(features)
    return estimator.predict(transfomed_features)[0]


def main():
    st.set_page_config(page_title="Students Perfomance Metrics")
    st.header("Students Perfomance Metrics")
    with st.form(key="form1"):
        col1, col2 = st.columns((2,2))
        with col1:
            gender = st.selectbox("Gender", ['female', 'male'])
            race_ethnicity = st.selectbox("Race Ethnicity",['group B', 'group C', 'group A', 'group D', 'group E'])
            parental_level_of_education = st.selectbox("Parental Education", ["bachelor's degree", 'some college', "master's degree","associate's degree", 'high school', 'some high school'])
            lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
        with col2:
            test_preparation_course = st.selectbox("Course", ['none', 'completed'])
            reading_score = st.number_input("Reading Score", min_value=0, max_value=100)
            writing_score = st.number_input("Writing Score", min_value=0, max_value=100)
        if st.form_submit_button("Predict",use_container_width=True):
            inputs = CustomData(gender,race_ethnicity,
                                parental_level_of_education,
                                lunch, test_preparation_course,
                                reading_score, writing_score)
            features_as_frame : pd.DataFrame = inputs.get_data_as_frame()
            result = predict_value(features_as_frame)
            st.markdown(
                f"""
                    ### Math score is approximately : {result}"""
            )
            

if __name__ == "__main__":
    main()

