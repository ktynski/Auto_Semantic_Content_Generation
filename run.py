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

import json
openai.api_key = openai.api_key = os.environ['openai_api_key']



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
def scrape_google(query):
    # Set headers to mimic a web browser
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    # Build the URL with the query parameter
    url = f'https://www.google.com/search?q={query}'
    # Send a request to the URL and store the HTML content
    html = requests.get(url, headers=headers).content
    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')
    # Find all the search result elements
    search_results = soup.find_all('div', {'class': 'g'})
    # Initialize an empty list to store the search results
    results = []
    # Loop through each search result and extract the relevant information
    for result in search_results:
        try:
            title = result.find('h3').text
            url = result.find('a')['href']
            results.append((title, url))
        except:
            continue
    # Create a dataframe from the search results
    df = pd.DataFrame(results, columns=['Title', 'URL'])
    df.to_csv("Scraped_URLs_From_SERPS.csv")
    return df


def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""



# Define a function to perform NLP analysis and return a string of keyness results
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
    return results_str

# Define the main function to scrape Google search results and analyze the article text
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
    df.to_csv("NLP_Data_On_SERP_Links_Text.csv")
    return df



# Define the main function to scrape Google search results and analyze the article text
def analyze_serps(query):
    # Scrape Google search results and create a dataframe
    df = scrape_google(query)
    # Scrape article text for each search result and store it in the dataframe
    for index, row in df.iterrows():
        url = row['URL']
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
        df.at[index, 'Most Common Words'] = ', '.join([word[0] for word in most_common])
        df.at[index, 'Least Common Words'] = ', '.join([word[0] for word in least_common])
        df.at[index, 'Most Common Bigrams'] = ', '.join([f'{bigram[0]} {bigram[1]}' for bigram in bigrams])
        df.at[index, 'Most Common Trigrams'] = ', '.join([f'{trigram[0]} {trigram[1]} {trigram[2]}' for trigram in trigrams])
        df.at[index, 'Most Common Quadgrams'] = ', '.join([f'{quadgram[0]} {quadgram[1]} {quadgram[2]} {quadgram[3]}' for quadgram in quadgrams])
        df.at[index, 'POS Tags'] = ', '.join([f'{token}/{tag}' for token, tag in pos_tags])
        # Replace any remaining commas with spaces in the Article Text column
        df.at[index, 'Article Text'] = ' '.join(row['Article Text'].replace(',', ' ').split())
    # Save the final dataframe as an Excel file
    writer = pd.ExcelWriter('NLP_Based_SERP_Results.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()
    # Return the final dataframe
    return df




# Define a function to summarize the NLP results from the dataframe
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
    summary += f'Median words per article: {median_words}\n'
    summary += f'Most common words: {top_words} ({len(word_freqs)} total words)\n'
    summary += f'Most common bigrams: {top_bigrams} ({len(bigram_freqs)} total bigrams)\n'
    summary += f'Most common trigrams: {top_trigrams} ({len(trigram_freqs)} total trigrams)\n'
    summary += f'Most common quadgrams: {top_quadgrams} ({len(quadgram_freqs)} total quadgrams)\n'
    return summary





def save_to_file(filename, content):
    with open(filename, 'w') as f:
        f.write("\n".join(content))

def generate_content(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
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
    #print(response)
    return response.strip().split('\n')

def generate_semantic_improvements_guide(prompt,query, model="gpt-3.5-turbo", max_tokens=2000, temperature=0.4):
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at Semantic SEO. In particular, you are superhuman at taking the result of an NLP keyword analysis of a search engine results page for a given keyword, and using it to build a readout/guide that can be used to inform someone writing a long-form article about a given topic so that they can best fully cover the semantic SEO as shown in the SERP. The goal of this guide is to help the writer make sure that the content they are creating is as comprehensive to the semantic SEO expressed in the content that ranks on the first page of Google for the given query. With the following semantic data, please provide this readout/guide. This readout/guide should be useful to someone writing about the topic, and should not include instructions to add info to the article about the SERP itself. The SERP semantic SEO data is just to be used to help inform the guide/readout. Please provide the readout/guide in well organized and hierarchical markdown."},
            {"role": "user", "content": f"Semantic SEO data for the keyword based on the content that ranks on the first page of google for the given keyword query of: {query} and it's related semantic data:  {prompt}"}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    #print(response)
    formatted_response = response.strip().split('\n')
    save_to_file("Semantic_SEO_Readout.txt", formatted_response)
    return str(response)   

def generate_outline(topic, model="gpt-3.5-turbo", max_tokens=1000):
    prompt = f"Generate an incredibly thorough article outline for the topic: {topic}. Consider all possible angles and be as thorough as possible. Please use Roman Numerals for each section."
    outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    save_to_file("outline.txt", outline)
    return outline

def improve_outline(outline, semantic_readout, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Given the following article outline, please improve and extend this outline significantly as much as you can keeping in mind the SEO keywords and data being provided in our semantic seo readout. Do not include a section about semantic SEO itself, you are using the readout to better inform your creation of the outline. Try and include and extend this as much as you can. Please use Roman Numerals for each section. The goal is as thorough, clear, and useful out line as possible exploring the topic in as much depth as possible. Think step by step before answering. Please take into consideration the semantic seo readout provided here: {semantic_readout} which should help inform some of the improvements you can make, though please also consider additional improvements not included in this semantic seo readout.  Outline to improve: {outline}."
    improved_outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    save_to_file("improved_outline.txt", improved_outline)
    return improved_outline



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
        prompt = full_outline + specific_section + ", please write a thorough section that goes in-depth, provides detail and evidence, and adds as much additional value as possible. Keep whatever hierarchy you find. Section text:"
        section = generate_content(prompt, model=model, max_tokens=max_tokens)
        sections.append(section)
        save_to_file(f"section_{i+1}.txt", section)
    return sections


def improve_section(section, i, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Given the following section of the article: {section}, please make thorough and improvements to this section. Keep whatever hierarchy you find. Only provide the updated section, not the text of your recommendation, just make the changes. Provide the updated section in valid Markdown please. Updated Section with improvements:"
    improved_section = generate_content(prompt, model=model, max_tokens=max_tokens)
    save_to_file(f"improved_section_{i+1}.txt", improved_section)
    return " ".join(improved_section)  # join the lines into a single string






def concatenate_files(file_names, output_file_name):
    final_draft = ''
    
    for file_name in file_names:
        with open(file_name, 'r') as file:
            final_draft += file.read() + "\n\n"  # Add two newline characters between sections

    with open(output_file_name, 'w') as output_file:
        output_file.write(final_draft)

    print("Final draft created.\n")
    return final_draft



def generate_article(topic, model="gpt-3.5-turbo", max_tokens_outline=2000, max_tokens_section=2000, max_tokens_improve_section=4000):
    query = topic
    results = analyze_serps(query)
    summary = summarize_nlp(results)

    semantic_readout = generate_semantic_improvements_guide(topic, summary,  model=model, max_tokens=max_tokens_outline)
    

    print(f"Topic: {topic}\n")

    print(f"Semantic SEO Readout:")
    #display(Markdown(str(semantic_readout)))

    print("Generating initial outline...")
    initial_outline = generate_outline(topic, model=model, max_tokens=max_tokens_outline)
    print("Initial outline created.\n")

    print("Improving the initial outline...")
    improved_outline = improve_outline(initial_outline, semantic_readout, model='gpt-4', max_tokens=3000)
    print("Improved outline created.\n")

    print("Generating sections based on the improved outline...")
    sections = generate_sections(improved_outline, model=model, max_tokens=max_tokens_section)
    print("Sections created.\n")

    print("Improving sections...")
    file_names = [f"improved_section_{i+1}.txt" for i in range(len(sections))]
    for i, section in enumerate(sections):
        improve_section(section, i, model='gpt-4', max_tokens=3000)
    print("Improved sections created.\n")

    print("Creating final draft...")
    final_draft = concatenate_files(file_names, "final_draft.txt")
    #display(Markdown(final_draft))
    return final_draft
    



def main():
    topic = st.text_input("Enter topic:", "Digital PR tips to earn media coverage in 2023")
    if st.button('Generate Content'):
        with st.spinner("Generating content..."):
            final_draft = generate_article(topic)  # rename your main function to generate_article
            st.markdown(final_draft)

if __name__ == "__main__":
    main()







