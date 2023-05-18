# Import necessary libraries
import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import newspaper
from newspaper import Article
import nltk
import statistics
import collections
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.collocations import QuadgramAssocMeasures, QuadgramCollocationFinder
import time
import openai
import pandas as pd
import re
import streamlit as st
from apify_client import ApifyClient
import pandas as pd
import transformers
from transformers import GPT2Tokenizer

import json
#openai.api_key = openai.api_key = os.environ['openai_api_key']
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('vader_lexicon')
nltk.download('inaugural')
nltk.download('webtext')
nltk.download('treebank')
nltk.download('gutenberg')
nltk.download('genesis')
nltk.download('trigram_collocations')
nltk.download('quadgram_collocations')


# Define a function to scrape Google search results and create a dataframe
from apify_client import ApifyClient
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def scrape_google(search):
    # Define the Apify API URL and the actor's name
    APIFY_API_URL = 'https://api.apify.com/v2'
    ACTOR_NAME = 'apify/google-search-scraper'

    # Retrieve the Apify API key from Streamlit secrets
    APIFY_API_KEY = st.secrets["APIFY_API_KEY"]

    # Initialize the ApifyClient with your API token
    client = ApifyClient(APIFY_API_KEY)

    # Prepare the actor input
    run_input = {
        "csvFriendlyOutput": False,
        "customDataFunction": "async ({ input, $, request, response, html }) => {\n  return {\n    pageTitle: $('title').text(),\n  };\n};",
        "includeUnfilteredResults": False,
        "maxPagesPerQuery": 1,
        "mobileResults": False,
        "queries": search,
        "resultsPerPage": 10,
        "saveHtml": False,
        "saveHtmlToKeyValueStore": False
    }

    print(f"Running Google Search Scrape for {search}")
    # Run the actor and wait for it to finish
    run = client.actor(ACTOR_NAME).call(run_input=run_input)
    print(f"Finished Google Search Scrape for {search}")

    # Fetch the actor results from the run's dataset
    results = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        results.append(item)

    # Extract URLs from organic results
    organic_results = [item['organicResults'] for item in results]
    urls = [result['url'] for sublist in organic_results for result in sublist]

    # Create DataFrame
    df = pd.DataFrame(urls, columns=['url'])

    # Print the dataframe
    print(df)
    st.header("Scraped Data from SERP and SERP Links")
    #st.write(df)
    return df



@st.cache_data(show_spinner=False)
def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""



@st.cache_data(show_spinner=False)
def truncate_to_token_length(input_string, max_tokens=1700):
    # Tokenize the input string
    tokens = tokenizer.tokenize(input_string)
    
    # Truncate the tokens to a maximum of max_tokens
    truncated_tokens = tokens[:max_tokens]
    
    # Convert the truncated tokens back to a string
    truncated_string = tokenizer.convert_tokens_to_string(truncated_tokens)
    
    return truncated_string


# Define a function to perform NLP analysis and return a string of keyness results

@st.cache_data(show_spinner=False)
def analyze_text(text):
    # Tokenize the text and remove stop words
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stopwords.words('english')]
    # Get the frequency distribution of the tokens
    fdist = FreqDist(tokens)
    # Create a bigram finder and get the top 20 bigrams by keyness
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    bigrams = finder.nbest(bigram_measures.raw_freq, 20)
    # Create a string from the keyness results
    results_str = ''
    results_str += 'Top 20 Words:\n'
    for word, freq in fdist.most_common(20):
        results_str += f'{word}: {freq}\n'
    results_str += '\nTop 20 Bigrams:\n'
    for bigram in bigrams:
        results_str += f'{bigram[0]} {bigram[1]}\n'
    st.write(results_str)    
    return results_str

# Define the main function to scrape Google search results and analyze the article text

@st.cache_data(show_spinner=False)
def main(query):
    # Scrape Google search results and create a dataframe
    df = scrape_google(query)
    # Scrape article text for each search result and store it in the dataframe
    for index, row in df.iterrows():
        url = row['URL']
        article_text = scrape_article(url)
        df.at[index, 'Article Text'] = article_text
    # Analyze the article text for each search result and store the keyness results in the dataframe
    for index, row in df.iterrows():
        text = row['Article Text']
        keyness_results = analyze_text(text)
        df.at[index, 'Keyness Results'] = keyness_results
    # Return the final dataframe
    #df.to_csv("NLP_Data_On_SERP_Links_Text.csv")
    return df



# Define the main function to scrape Google search results and analyze the article text

@st.cache_data(show_spinner=False)
def analyze_serps(query):
    # Scrape Google search results and create a dataframe
    df = scrape_google(query)
    # Scrape article text for each search result and store it in the dataframe
    for index, row in df.iterrows():
        url = row['url']
        #st.write(url)
        article_text = scrape_article(url)
        df.at[index, 'Article Text'] = article_text
    # Analyze the article text for each search result and store the NLP results in the dataframe
    for index, row in df.iterrows():
        text = row['Article Text']
        # Tokenize the text and remove stop words
        tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stopwords.words('english') and 'contact' not in word.lower() and 'admin' not in word.lower()]
        # Calculate the frequency distribution of the tokens
        fdist = FreqDist(tokens)
        # Calculate the 20 most common words
        most_common = fdist.most_common(20)
        # Calculate the 20 least common words
        least_common = fdist.most_common()[-20:]
        # Calculate the 20 most common bigrams
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokens)
        bigrams = finder.nbest(bigram_measures.raw_freq, 20)
        # Calculate the 20 most common trigrams
        trigram_measures = TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(tokens)
        trigrams = finder.nbest(trigram_measures.raw_freq, 20)
        # Calculate the 20 most common quadgrams
        quadgram_measures = QuadgramAssocMeasures()
        finder = QuadgramCollocationFinder.from_words(tokens)
        quadgrams = finder.nbest(quadgram_measures.raw_freq, 20)
        # Calculate the part-of-speech tags for the text
        pos_tags = nltk.pos_tag(tokens)
        # Store the NLP results in the dataframe
        df.at[index, "Facts"] = generate_content3(text)
        df.at[index, 'Most Common Words'] = ', '.join([word[0] for word in most_common])
        df.at[index, 'Least Common Words'] = ', '.join([word[0] for word in least_common])
        df.at[index, 'Most Common Bigrams'] = ', '.join([f'{bigram[0]} {bigram[1]}' for bigram in bigrams])
        df.at[index, 'Most Common Trigrams'] = ', '.join([f'{trigram[0]} {trigram[1]} {trigram[2]}' for trigram in trigrams])
        df.at[index, 'Most Common Quadgrams'] = ', '.join([f'{quadgram[0]} {quadgram[1]} {quadgram[2]} {quadgram[3]}' for quadgram in quadgrams])
        df.at[index, 'POS Tags'] = ', '.join([f'{token}/{tag}' for token, tag in pos_tags])
        # Replace any remaining commas with spaces in the Article Text column
        df.at[index, 'Article Text'] = ' '.join(row['Article Text'].replace(',', ' ').split())
    # Save the final dataframe as an Excel file
    #writer = pd.ExcelWriter('NLP_Based_SERP_Results.xlsx', engine='xlsxwriter')
    #df.to_excel(writer, sheet_name='Sheet1', index=False)
    #writer.save()
    st.write(df)
    # Return the final dataframe
    return df




# Define a function to summarize the NLP results from the dataframe


@st.cache_data(show_spinner=False)
def summarize_nlp(df):
    # Calculate the total number of search results
    total_results = len(df)
    # Calculate the average length of the article text
    avg_length = round(df['Article Text'].apply(len).mean(), 2)
    # Get the most common words across all search results
    all_words = ', '.join(df['Most Common Words'].sum().split(', '))
    # Get the most common bigrams across all search results
    all_bigrams = ', '.join(df['Most Common Bigrams'].sum().split(', '))
    # Get the most common trigrams across all search results
    all_trigrams = ', '.join(df['Most Common Trigrams'].sum().split(', '))
    # Get the most common quadgrams across all search results
    all_quadgrams = ', '.join(df['Most Common Quadgrams'].sum().split(', '))
    # Get the most common part-of-speech tags across all search results
    all_tags = ', '.join(df['POS Tags'].sum().split(', '))
    # Calculate the median number of words in the article text
    median_words = statistics.median(df['Article Text'].apply(lambda x: len(x.split())).tolist())
    # Calculate the frequency of each word across all search results
    word_freqs = collections.Counter(all_words.split(', '))
    # Calculate the frequency of each bigram across all search results
    bigram_freqs = collections.Counter(all_bigrams.split(', '))
    # Calculate the frequency of each trigram across all search results
    trigram_freqs = collections.Counter(all_trigrams.split(', '))
    # Calculate the frequency of each quadgram across all search results
    quadgram_freqs = collections.Counter(all_quadgrams.split(', '))
    # Calculate the top 20% of most frequent words
    top_words = ', '.join([word[0] for word in word_freqs.most_common(int(len(word_freqs) * 0.2))])
    # Calculate the top 20% of most frequent bigrams
    top_bigrams = ', '.join([bigram[0] for bigram in bigram_freqs.most_common(int(len(bigram_freqs) * 0.2))])
    # Calculate the top 20% of most frequent trigrams
    top_trigrams = ', '.join([trigram[0] for trigram in trigram_freqs.most_common(int(len(trigram_freqs) * 0.2))])
    # Calculate the top 20% of most frequent quadgrams
    top_quadgrams = ', '.join([quadgram[0] for quadgram in quadgram_freqs.most_common(int(len(quadgram_freqs) * 0.2))])
    
    #print(f'Total results: {total_results}')
    #print(f'Average article length: {avg_length} characters')
    #print(f'Median words per article: {median_words}')
    #print(f'Most common words: {top_words} ({len(word_freqs)} total words)')
    #print(f'Most common bigrams: {top_bigrams} ({len(bigram_freqs)} total bigrams)')
    #print(f'Most common trigrams: {top_trigrams} ({len(trigram_freqs)} total trigrams)')
    #print(f'Most common quadgrams: {top_quadgrams} ({len(quadgram_freqs)} total quadgrams)')
    #print(f'Most common part-of-speech tags: {all_tags}')
    summary = ""
    summary += f'Total results: {total_results}\n'
    summary += f'Average article length: {avg_length} characters\n'
    summary += f'Average article length: {avg_length} characters\n'
    summary += f'Median words per article: {median_words}\n'
    summary += f'Most common words: {top_words} ({len(word_freqs)} total words)\n'
    summary += f'Most common bigrams: {top_bigrams} ({len(bigram_freqs)} total bigrams)\n'
    summary += f'Most common trigrams: {top_trigrams} ({len(trigram_freqs)} total trigrams)\n'
    summary += f'Most common quadgrams: {top_quadgrams} ({len(quadgram_freqs)} total quadgrams)\n'
    #summary = '\n'.join(summary)
    #st.markdown(str(summary))
    return summary





#def save_to_file(filename, content):
    #with open(filename, 'w') as f:
        #f.write("\n".join(content))


@st.cache_data(show_spinner=False)
def generate_content(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    #st.write(prompt)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simulate an exceptionally talented journalist and editor. Given the following instructions, think step by step and produce the best possible output you can."},
            {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response.strip().split('\n')

        #except:
            #st.write(f"Attempt {i+1} failed, retrying...")
            #time.sleep(3)  # Wait for 3 seconds before next try

    #st.write("OpenAI is currently overloaded, please try again later.")
    #return None

@st.cache_data(show_spinner=False)
def generate_content2(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    #st.write(prompt)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simulate an exceptionally talented journalist and editor. Given the following instructions, think step by step and produce the best possible output you can. Return the results in Nicely formatted markdown please."},
            {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response

        #except:
            #st.write(f"Attempt {i+1} failed, retrying...")
            #time.sleep(3)  # Wait for 3 seconds before next try

    #st.write("OpenAI is currently overloaded, please try again later.")
    #return None

    
@st.cache_data(show_spinner=False)
def generate_content3(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    #st.write(prompt)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simulate an exceptionally talented investigative journalist and researcher. Given the following text, please write a short paragraph providing only the most important facts and takeaways that can be used later when writing a full analysis or article."},
            {"role": "user", "content": f"Use the following text to provide the readout: {prompt}"}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response    
    
    
    
@st.cache_data(show_spinner=False)
def generate_semantic_improvements_guide(prompt,query, model="gpt-3.5-turbo", max_tokens=2000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,1500)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are an expert at Semantic SEO. In particular, you are superhuman at taking  a given NLTK report on a given text corpus compiled from the text of the linked pages returned for a google search.
            and using it to build a comprehensive set of instructions for an article writer that can be used to inform someone writing a long-form article about a given topic so that they can best fully cover the semantic SEO as shown in NLTK data from the SERP corpus. 
             Provide the result in well formatted markdown. The goal of this guide is to help the writer make sure that the content they are creating is as comprehensive to the semantic SEO with a focus on what is most imprtant from a semantic SEO perspective."""},
            {"role": "user", "content": f"Semantic SEO data for the keyword based on the content that ranks on the first page of google for the given keyword query of: {query} and it's related semantic data:  {prompt}"}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    st.header("Semantic Improvements Guide")
    st.markdown(response,unsafe_allow_html=True)
    return str(response)

        #except:
            #st.write(f"Attempt {i+1} failed, retrying...")
            #time.sleep(3)  # Wait for 3 seconds before next try

    #st.write("OpenAI is currently overloaded, please try again later.")
    #return None
    
   

@st.cache_data(show_spinner=False)
def generate_outline(topic, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Generate an incredibly thorough article outline for the topic: {topic}. Consider all possible angles and be as thorough as possible. Please use Roman Numerals for each section."
    outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    #save_to_file("outline.txt", outline)
    return outline

@st.cache_data(show_spinner=False)
def improve_outline(outline, semantic_readout, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Given the following article outline, please improve and extend this outline significantly as much as you can keeping in mind the SEO keywords and data being provided in our semantic seo readout. Do not include a section about semantic SEO itself, you are using the readout to better inform your creation of the outline. Try and include and extend this as much as you can. Please use Roman Numerals for each section. The goal is as thorough, clear, and useful out line as possible exploring the topic in as much depth as possible. Think step by step before answering. Please take into consideration the semantic seo readout provided here: {semantic_readout} which should help inform some of the improvements you can make, though please also consider additional improvements not included in this semantic seo readout.  Outline to improve: {outline}."
    improved_outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    #save_to_file("improved_outline.txt", improved_outline)
    return improved_outline



@st.cache_data(show_spinner=False)
def generate_sections(improved_outline, model="gpt-3.5-turbo", max_tokens=2000):
    sections = []

    # Parse the outline to identify the major sections
    major_sections = []
    current_section = []
    for part in improved_outline:
        if re.match(r'^[ \t]*[#]*[ \t]*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)\b', part):
            if current_section:  # not the first section
                major_sections.append('\n'.join(current_section))
                current_section = []
        current_section.append(part)
    if current_section:  # Append the last section
        major_sections.append('\n'.join(current_section))

    # Generate content for each major section
    for i, section_outline in enumerate(major_sections):
        full_outline = "Given the full improved outline: "
        full_outline += '\n'.join(improved_outline)
        specific_section = ", and focusing specifically on the following section: "
        specific_section += section_outline
        prompt =  specific_section + ", please write a thorough section that goes in-depth, provides detail and evidence, and adds as much additional value as possible. Keep whatever hierarchy you find. Never write a conclusion part of a section unless the section itself is supposed to be a conclusion. Section text:"
        section = generate_content(prompt, model=model, max_tokens=max_tokens)
        sections.append(section)
        #save_to_file(f"section_{i+1}.txt", section)
    return sections

@st.cache_data(show_spinner=False)
def improve_section(section, i, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Given the following section of the article: {section}, please make thorough and improvements to this section. Keep whatever hierarchy you find. Only provide the updated section, not the text of your recommendation, just make the changes. Always provide the updated section in valid Markdown please. Updated Section with improvements:"
    prompt = str(prompt)
    improved_section = generate_content2(prompt, model=model, max_tokens=max_tokens)
    #st.markdown(improved_section)
    st.markdown(improved_section,unsafe_allow_html=True)
    return " ".join(improved_section)  # join the lines into a single string






@st.cache_data(show_spinner=False)
def concatenate_files(file_names, output_file_name):
    final_draft = ''
    
    for file_name in file_names:
        with open(file_name, 'r') as file:
            final_draft += file.read() + "\n\n"  # Add two newline characters between sections

    with open(output_file_name, 'w') as output_file:
        output_file.write(final_draft)

    #print("Final draft created.\n")
    return final_draft



@st.cache_data(show_spinner=False)
def generate_article(topic, model="gpt-3.5-turbo", max_tokens_outline=2000, max_tokens_section=2000, max_tokens_improve_section=4000):
    status = st.empty()
    status.text('Analyzing SERPs...')
    
    query = topic
    results = analyze_serps(query)
    summary = summarize_nlp(results)

    status.text('Generating semantic SEO readout...')
    semantic_readout = generate_semantic_improvements_guide(topic, summary,  model=model, max_tokens=max_tokens_outline)
    
    
    status.text('Generating initial outline...')
    initial_outline = generate_outline(topic, model=model, max_tokens=max_tokens_outline)

    status.text('Improving the initial outline...')
    improved_outline = improve_outline(initial_outline, semantic_readout, model=model, max_tokens=1500)
    #st.markdown(improved_outline,unsafe_allow_html=True)
    
    status.text('Generating sections based on the improved outline...')
    sections = generate_sections(improved_outline, model=model, max_tokens=max_tokens_section)

    status.text('Improving sections...')
    
    improved_sections = []
    for i, section in enumerate(sections):
        section_string = '\n'.join(section)
        status.text(f'Improving section {i+1} of {len(sections)}...')
        time.sleep(5)
        improved_sections.append(improve_section(section_string, i, model=model, max_tokens=1200))



    status.text('Finished')
    final_content = '\n'.join(improved_sections)
    #st.markdown(final_content,unsafe_allow_html=True)
   




def main():
    st.title('Long-form Article Generator with Semantic SEO Understanding')
    
    st.markdown('''
    Welcome to the long-form article generator! This application leverages advanced AI to create comprehensive articles based on the topic you provide. 

    Not only does it generate articles, but it also includes a Semantic SEO understanding. This means it takes into consideration the semantic context and relevance of your topic, based on current search engine results.

    Just input your topic below and let the AI do its magic!
    
    ** If you get an error, (sometimes OpenAI will be overloaded and not work), just press generate again and it should start where it left off.
    ''')
   
    topic = st.text_input("Enter topic:", "Digital PR tips to earn media coverage in 2023")

    # Get user input for API key
    user_api_key = st.text_input("Enter your OpenAI API key")

    if st.button('Generate Content'):
        if user_api_key:
            openai.api_key = user_api_key
            with st.spinner("Generating content..."):
                final_draft = generate_article(topic)
                #st.markdown(final_draft)
        else:
            st.warning("Please enter your OpenAI API key above.")

if __name__ == "__main__":
    main()







