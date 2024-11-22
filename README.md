# Supply-Chain-Case-Study-LDA-Topic-Modelling
This is the Streamlit dashboard to showcase the project of applying LDA Topic Modelling on Supply Chain Case Studies

This research project utilizes the application of Latent Dirichlet Allocation (LDA) topic modelling to a corpus of supply chain case study articles. 
This research shows that LDA is an effective tool for analyzing large volumes of textual data which offers different views of complex supply chain issues.

This project started from
1. Data collection to obtain 565 documents of case study from the SCOPUS, then using PyMUPDF to do text extraction and store in dataframe. 
2. Data preprocessing using SPACY includes stopwords removal, lemmatization, POS tag with only ["noun", "adj", "verb", "adverb"].
3. Topic Modeling with Gensim to get the Bags of Words(BoW) and fit in Gensim.LDA model.
4. Fine tuning the parameters of model (number of topics, alpha, beta, chunksize, passes).
5. The best 6 topics were extracted which represented different areas such as sustainability practices with supplier, capacity and transportation optimization, risk, food and agricultural, environmental sustainability and technological advancements.
6. The evolution of each topics also shown in the dashboard.
7. Significant topics in each individual case studies can be identified shown in dashboard.

This approach not only aids researchers in articles research on supply chain management but also provides practitioners with valuable information to inform strategic decision-making. 

Streamlit App: https://supply-chain-case-study-lda-topic-modelling-whanishere.streamlit.app/ 

![image](https://github.com/user-attachments/assets/9c921072-7554-486e-b4a2-566a9df2ce82)
