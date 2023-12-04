# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:02:08 2023

@author: Malshan Gunasekara

#streamlit run titanic_app.py 
"""
import streamlit as st
import streamlit.components.v1 as com
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers 
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras.models import load_model

data = pd.read_csv('./data/train.csv')

st.set_page_config(layout="wide")

with open("./css/style.css") as source:
    design= source.read().replace('\n', '')

html_code =f"""
        <style>
            {design}
        </style>

        <div>
            <h1 class="heading">The Titanic Tragedy</h1>
            <p class="quote">"The Titanic taught us that even the strongest can face unexpected challenges. It reminds us to be careful and prepared, no matter how confident we feel. Life is unpredictable, and we must always stay humble and ready for anything."
            </p>
            <p class="p1">Titanic disaster is a well known tragedy in world history. The Titanic was officially known as the RMS Titanic, indicating its status as a Royal Mail Ship. The Titanic was run by the White Star Line and was famous for being luxurious and having advanced engineering.
                        Ship tragically sank on April 15, 1912, during its maiden voyage from Southampton to New York City.
                        The ship hit an iceberg in the North Atlantic, which damaged it a lot. Even though they tried to get people off the ship, over 1,500 people died because there weren't enough lifeboats. The Titanic sinking is a strong symbol showing that being too sure about things can be dangerous. It also reminds us how important it is to have really good safety rules in the shipping industry.
            </p>
            <hr>
        </div>   
    """

st.markdown(html_code, unsafe_allow_html=True) 


def sidebar_bg(bg,css_class):

   side_bg_ext = '.jpg'

   st.markdown(
      f"""
      <style>
      [data-testid={css_class}] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

bg = './images/ship.jpg'
css1 = "block-container"
# css2 = "stVerticalBlock"
sidebar_bg(bg,css1)
# sidebar_bg(bg,css2)


col1_1, col1_2= st.columns([1,1])

with col1_1:
    st.header("Passenger Information Section")

    labels1 = data['Sex'].unique()
    values1 = [data['Sex'].value_counts()[0],data['Sex'].value_counts()[1]]

    cls = data['Pclass'].value_counts().to_dict()
    labels2 = list(cls.keys())
    values2 = list(cls.values())

    emb = data['Embarked'].value_counts().to_dict()
    labels3 = list(emb.keys())
    values3 = list(emb.values())

    age_hist = px.histogram(x=data['Age'], nbins=10)


    fig = sp.make_subplots(rows=2, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}], [{'type': 'xy'}, {'type': 'pie'}]],
                    subplot_titles=['Gender Distribution', 'Passenger Class', 'Age of Passengers', 'Passenger Embarked'])
    # Add the first Pie chart to the first subplot in the first row
    fig.add_trace(go.Pie(labels=labels1, values=values1, hole=0.3, name='Gender', legendgroup='gender_legend'), row=1, col=1)

    # Add the second Pie chart to the second subplot in the first row
    fig.add_trace(go.Pie(labels=labels2, values=values2, hole=0.3, name='Passenger Class', legendgroup='class_legend'), row=1, col=2)

    fig.add_trace(go.Pie(labels=labels3, values=values3, hole=0.3, name='Embarked Port', legendgroup='emb_legend'), row=2, col=2)

    fig.add_trace(go.Histogram(x=data['Age'], histnorm='percent', marker=dict(color='blue'), opacity=0.7, legendgroup='age_legend', name='Age Distribution'), row=2, col=1)

    fig.update_layout(title_text='Analysis of Titanic Passengers Data', showlegend=True)

    st.plotly_chart(fig)


with col1_2:
    st.header("Timeline of the Disaster")

    # Sample data for the Titanic timeline
    timeline_data = pd.DataFrame({
        'Event': ['Embarked Southampton', 'Embarked Cherbourg', 'Embarked Queenstown', 'Departure', 'Sinking', 'Rescue'],
        'Date': ['1912-04-10', '1912-04-10', '1912-04-11', '1912-04-10', '1912-04-15', '1912-04-15'],
        'Description': ['Passengers embark at Southampton', 'Passengers embark at Cherbourg', 'Passengers embark at Queenstown',
                        'Titanic departs Southampton', 'Titanic sinks', 'Rescue operations']
    })

    # Convert the 'Date' column to datetime
    timeline_data['Date'] = pd.to_datetime(timeline_data['Date'])

    # Sort the dataframe by date
    timeline_data = timeline_data.sort_values(by='Date')

    # Create the timeline line chart
    fig = px.line(timeline_data, x='Date', y='Event', text='Description',
                title='Titanic Timeline', labels={'Event': 'Event on Timeline'},
                template='plotly_white', color_discrete_sequence=['blue'])

    # Update layout for better timeline appearance
    fig.update_traces(mode='markers+lines', marker=dict(size=12, symbol='circle'), textposition='bottom center')
    fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title=''),
                    margin=dict(l=0, r=0, t=50, b=0), showlegend=False)

    # Display the figure using st.plotly_chart
    st.plotly_chart(fig)



#load the previously saved model
dl_model = load_model("./model/DLmodel")

# ---------------------------------------------
# Initialize session state
if 'show_form' not in st.session_state:
    st.session_state.show_form = False

# Button to toggle the prediction form visibility
show_form_button = st.button("Predict Survival")

# Reset button to clear form values -----------------------
# reset_button = st.button("Reset Form")

if show_form_button:
    st.session_state.show_form = True

# Main app logic
if st.session_state.show_form:
    with st.form("prediction_form",clear_on_submit=True):

        # Create two columns
        col1, col2, col3 = st.columns([1,1,2])

        # First column
        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, step=1, value=0)
            Gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            class_type = st.number_input("Class", min_value=1, max_value=3, step=1, value=1)
            fare = st.number_input("Fare", min_value=0.00, max_value=550.00, step=0.25, value=0.00)            

        # Second column
        with col2:
            parch = st.number_input("Parent/children", min_value=0, max_value=10, step=1, value=0)
            sibSp = st.number_input("Siblings/Spouse", min_value=0, max_value=10, step=1, value=0)
            embarked_mapping = {
                'Cherbourg': 'C',
                'Queenstown': 'Q',
                'Southampton': 'S'
            }
            embarked_selected = st.selectbox("Embarked", embarked_mapping.keys(), index=0)
            embarked = embarked_mapping.get(embarked_selected, 'Unknown')

            
        # Third column
        with col3:
            st.write(" ")
            st.write(" ")
            st.markdown("""<p style="font-size:17px; color:#2255ff;"><b>Number of parents or children a passenger had aboard the Titanic</b></p>""",unsafe_allow_html=True)
            st.write(" ")
            st.write(" ")
            st.write("""<p style="font-size:17px; color:#2255ff;"><b>Number of siblings or spouses that a passenger had aboard the Titanic</b></p>""",unsafe_allow_html=True)
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write("""<p style="font-size:17px; color:#2255ff;"><b>Port of embarkation, indicating the location where a passenger boarded the Titanic</b></p>""",unsafe_allow_html=True)
        

        submit_button = st.form_submit_button("Predict")

    # if reset_button:
    #     st.form.clear()

    if submit_button:

        # Manually create binary columns for gender and Embarked
        gender = 1 if Gender == "Male" else 0

        if (embarked == "C"): 
            Embarked_Q = 0
            Embarked_S = 0
        elif(embarked == "Q"):
            Embarked_Q = 1
            Embarked_S = 0
        elif(embarked == "S"):
            Embarked_Q = 0
            Embarked_S = 1

        age_min = min(data['Age'])
        age_max = max(data['Age'])

        age = (age-age_min)/(age_max-age_min)

        fare_min = min(data['Fare'])
        fare_max = max(data['Fare'])

        fare = (fare-fare_min)/(fare_max-fare_min)

        form_data = pd.DataFrame({
            "Age": [age],
            "Fare": [fare],
            "Pclass": [class_type],
            "Parch": [parch],
            "SibSp": [sibSp],
            "Sex_male": [gender],
            "Embarked_Q": [Embarked_Q],
            "Embarked_S": [Embarked_S],
        })

        predictions = dl_model.predict(form_data)

        if predictions >= 0.5:
            output='Survived'
            st.success(f"This passenger {output}")
        else:
            output='Not Survived'
            st.error(f"This passenger {output}")
            
