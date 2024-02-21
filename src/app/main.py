import streamlit as st
import sys
sys.path.append('../')
from app.pipelines import pipeline_inference
from app.xai import get_explanation
import time
import pandas as pd
import plotly.express as px


st.title('Text identification app')

st.subheader('This app is designed to identify if a text was written by a human or an AI')
st.markdown('In many cases, using AI is not a suitable solution because this does not allow to develop creativity and innovation in written assessments')

col1, col2 = st.columns(2)
with col1:
    a = st.button('Classify text')
with col2:
    xai_option = st.toggle('Explain the classification', value = False)

with st.sidebar:
    st.subheader('About the App')
    st.markdown('Data used for the training come from the following source: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text')
    st.markdown('The model built is not based on transformer architecture, it uses traditional Natural Language Processing techniques')
    st.empty()
    st.subheader('Author')
    st.markdown('Sebastián Sarasti Zambonino')
    st.markdown('Data Scientist - Machine Learning Engineer')
    st.markdown('https://www.linkedin.com/in/sebastiansarasti/')
    st.markdown('https://github.com/sebassaras02')

text_input = st.text_area('Enter the text to classify', height = 200)


result = None
if a and not xai_option:
    if text_input:
        with st.spinner('Classifying the text, wait please ...'):
            time.sleep(1)
        result = pipeline_inference(text_input)

        st.subheader('Probability that the text was classified as:')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Human written', result[0][0] )
        with col2:
            st.metric('AI written', result[0][1])
        if result[0][1]>0.6:
            st.warning('High probability that the text was written by an AI')
        else:
            st.success('High probability that the text was written by a human')
    else:
        st.exception('Please enter the text to classify, no text was provided')

elif a and xai_option:
    if text_input:
        with st.spinner('Classifying the text, wait please ...'):
            time.sleep(1)
        result = pipeline_inference(text_input)

        st.subheader('Probability that the text was classified as:')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Human written', result[0][0] )
        with col2:
            st.metric('AI written', result[0][1])
        if result[0][1]>0.6:
            st.warning('High probability that the text was written by an AI')
        else:
            st.success('High probability that the text was written by a human')
        
        with st.spinner('Explaining the classification, wait please ...'):
            explanation = get_explanation(text_input)
            df = pd.DataFrame(list(explanation.items()), columns=['Palabras', 'Números'])
            df['Signo'] = ['Positivo' if x >= 0 else 'Negativo' for x in df['Números']]
            df = df.sort_values('Números', ascending=False)
            df = df.rename(columns={'Palabras': 'Words', 'Números': 'Frequency', 'Signo': 'Type'})
            df['Type'] = df['Type'].map({'Positivo': 'IA Pattern', 'Negativo': 'Humman Pattern'})
            fig = px.bar(df, y='Words', x='Frequency', color='Type', color_discrete_map={'IA Pattern': 'red', 'Humman Pattern': 'blue'})
            st.subheader('Explanation of the classification:')
            st.markdown('The following words are the most important to classify the text:')
            st.plotly_chart(fig)

        

