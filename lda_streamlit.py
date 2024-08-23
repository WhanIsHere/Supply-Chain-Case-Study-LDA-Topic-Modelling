import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import streamlit.components.v1 as components


# Set page configuration
st.set_page_config(
    page_title="LDA Topic Modelling",
    page_icon=":bookmark_tabs:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Filepaths
lda_results_file = 'Streamlit Files/lda_topics.csv'
articles_significant_topics = 'Streamlit Files/articles_significant_topics.csv'
lda_params_file = 'Streamlit Files/lda_params.csv'
articles_file = 'Streamlit Files/cleaned_article.pkl'  
dictionary_file = 'Streamlit Files/dictionary_after.pkl'
html_path = 'Streamlit Files/lda.html'
word_cloud_path = 'Streamlit Files/WordCloud_Topic.png'
topic_evolution_image_path = 'Streamlit Files/Topic_Evolution.png'  

# Load data
lda_df = pd.read_csv(lda_results_file)
articles_df = pd.read_pickle(articles_file)
results_df = pd.read_csv(articles_significant_topics)  
lda_params_df = pd.read_csv(lda_params_file)
with open(dictionary_file, "rb") as file:
    dictionary_after = pickle.load(file)
    

# Title and Description
st.title("💡 Supply Chain Case Studies Topic Modelling")
st.subheader("This dashboard presents topic modeling results for supply chain case studies using the LDA technique.", divider='rainbow')

# Create tabs
tab1, tab2, tab3 = st.tabs(["**📜 Text Data**", "🎯 **LDA Results**", "**🗂️ Case Studies Clustering**"])

# Tab 1: Data
with tab1:

    st.header("📃 Case Studies Used")
    # Date Range
    st.write("**YEAR: 1998 ⟵⟶ 2024**")
    
    # Input Data Info
    number_of_papers = 565
    total_words = '3,508,721'
    total_preprocessed_words = '1,757,209'
    unique_words = '31,026 ⟶ 9,365'
    # Creating columns for each input data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="**📑 Number of Papers**", value=number_of_papers)
    with col2:
        st.metric(label="**Totals Words**", value=total_words)
    with col3:
        st.metric(label="**Total Words After Preprocessing**", value=total_preprocessed_words)
    with col4:
        st.metric(label="**Unique Words After Pruning**", value=unique_words)
        
    # Session state for selected article
    if 'selected_article' not in st.session_state:
        st.session_state['selected_article'] = None
    
    # Article selection
    st.header("Select Case Study")
    selected_article = st.selectbox("Choose a case study", articles_df["Article Name"])
    
    if selected_article != st.session_state['selected_article']:
        st.session_state['selected_article'] = selected_article
    
    # Display article details
    article_details = articles_df[articles_df["Article Name"] == selected_article].iloc[0]
    
    col1, col2 = st.columns(2)
    
    # Column 1: Show original article text
    with col1:
        with st.expander("Show Original Text"):
            st.write(article_details["Text"])
    
    # Column 2: Show preprocessed text
    with col2:
        with st.expander("Apply Preprocessing"):
            st.write(article_details["Preprocessed Text"])
    
    # Display unique dictionary words
    st.header("Unique Dictionary")
    st.write(f"Number of unique dictionary: {len(dictionary_after)}")
    with st.expander("Expand to see all unique dictionary"):
            st.write(dictionary_after)



# Tab 2: Result
with tab2:
    # Display the parameters of LDA model
    st.header("🎛️ Parameters of LDA Topic Model")
    # Convert the DataFrame to a dictionary and extract the first row
    lda_params = lda_params_df.iloc[0].to_dict()
    # Use columns to display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="**Number of Topics**", value=lda_params['Number of Topics'])
    with col2:
        st.metric(label="**Alpha**", value=lda_params['Alpha'])
    with col3:
        st.metric(label="**Beta**", value=lda_params['Beta'])
    with col4:
        st.metric(label="**Chunksize**", value=lda_params['Chunksize'])
    with col5:
        st.metric(label="**Passes**", value=lda_params['Passes'])

    st.header("Coherence Score")
    st.subheader("**Overall: 0.5121**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(label="**Topic 0**", value=0.4982)
    with col2:
        st.metric(label="**Topic 1**", value=0.5377)
    with col3:
        st.metric(label="**Topic 2**", value=0.3477)
    with col4:
        st.metric(label="**Topic 3**", value=0.5615)
    with col5:
        st.metric(label="**Topic 4**", value=0.6119)
    with col6:
        st.metric(label="**Topic 5**", value=0.5153)
    
    # Display Word Clouds for each topic
    st.header("Topics and Dominant Words")
    st.image(word_cloud_path, caption="WordClouds", use_column_width=True)
    
    # Display pyLDAvis HTML visualization
    with open(html_path, 'r', encoding='utf-8') as html:
        lda_html = html.read()
        st.header("PyLDAvis")
        components.html(lda_html, width=1200, height=900, scrolling=True)

    # Display Evolution of Topics Over Time
    st.header("Evolution of Topics Over Time")
    st.image(topic_evolution_image_path, caption="Evolution of Topics Over Time", use_column_width=True)



# Tab 3: Articles Clustering
with tab3:
    st.header("🔍 Topic Assigned")
    st.dataframe(articles_df[['Year', 'Article Name', 'Topic_Number', 'Topic_Probability']], width=1600)

    # Creating columns for each metric
    col1, col2 = st.columns(2)
    
    with col1:
        # Group by 'Topic_Number' and count the number of articles in each topic
        topic_counts = articles_df['Topic_Number'].value_counts().reset_index()
        topic_counts.columns = ['Topic_Number', 'Article_Count']
        topic_counts = topic_counts.rename(columns={'Topic_Number': 'Topic', 'Article_Count': 'Number of Case Study'})
        
        # Display Barchart of total number case study in each topic
        st.header("Number of Case Study per Topic")
        st.bar_chart(topic_counts, x='Topic', y='Number of Case Study' ,height=450 )
        
   
    with col2:

        st.header("Significant Topics within Case Study")
    
        # Select case study
        selected_article_name = st.selectbox(
            "Choose a case study", 
            results_df["Article Name"], 
            key="article_selectbox_tab3"
        )
    
        # Get the significant topics for the selected case study
        article_row = results_df[results_df["Article Name"] == selected_article_name].iloc[0]
        
        # Filter out columns
        significant_topics = article_row.drop(labels=["Article Name"])
        
        # Prepare the data for the bar chart
        significant_topics_df = pd.DataFrame({
            'Topic': significant_topics.index,
            'Significance': significant_topics.values
        })
             
        # Display bar graph for significant topics within each case study
        st.bar_chart(significant_topics_df.set_index('Topic'), y='Significance',height=380)

