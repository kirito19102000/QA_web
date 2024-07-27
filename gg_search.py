from googlesearch import search
import urllib.request
from bs4 import BeautifulSoup
import urllib.parse
import requests
from data import GetContext,GetSimilar
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import torch

def clean_text(content):
    text=content
    text=text.replace('\n\n\n', '\n')
    text=text.replace('\n\n', '\n')
    text=text.replace('• \n', '• ')
    text=text.replace('- \n', '- ')
    text=text.replace('• - ','- ')
    text=text.replace('• -', '-')
    return text


def google_search(query):
    query = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        return None

def parse_results(html):
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    for g in soup.find_all('div', class_='tF2Cxc'):
        title = g.find('h3').text
        link = g.find('a')['href']
        results.append({
            'title': title,
            'link': link
        })
    return results

def google_scrape(url):
    try:
        thepage = urllib.request.urlopen(url, timeout=10)
        soup = BeautifulSoup(thepage, "html.parser")
        return soup
    except Exception as e:
        return f"Error scraping {url}: {e}"

def split_content_by_h2(soup):
    sections = []
    current_section = []
    start_collecting = False

    for element in soup.find_all(['h2', 'h3', 'p', 'ul']):
        if element.name == 'h2':
            if start_collecting and current_section:
                sections.append('\n'.join(current_section))
                current_section = []
            start_collecting = True
            current_section.append(f"\n\n{element.get_text()}\n")
        elif element.name == 'h3':
            if start_collecting:
                # Add a section for h3 under the current h2
                current_section.append(element.get_text())
        elif element.name == 'p':
            if start_collecting:
                current_section.append(element.get_text())
        elif element.name == 'ul':
            if start_collecting:
                # Add each list item with a bullet point
                for li in element.find_all('li'):
                    current_section.append(f"• {li.get_text()}")

    if current_section:  # To add the last section if it exists
        sections.append('\n'.join(current_section))

    return sections

# query = 'nguyên nhân hàng đầu gây ra nám da là gì?'
# html = google_search(query)
# if html:
#     contexts=[]
#     search_results = parse_results(html)
#     for result in search_results:
#         print(f"Title: {result['title']}")
#         print(f"Link: {result['link']}")
#         soup = google_scrape(result['link'])
#         if isinstance(soup, str):  # Check if there was an error
#             print(soup)
#             continue
#         sections = split_content_by_h2(soup)
#         for i in range(len(sections)):
#             sections[i]=clean_text(sections[i])
#         for j in sections:
#             if len(j)<20:
#                 sections.remove(j)
#         contexts.append(GetSimilar(query,sections))
        
        
# else:
#     print("Failed to retrieve search results")

# print(GetSimilar(query,contexts))

# tokenizer = AutoTokenizer.from_pretrained("kirito546/QA-bert-multilingual")
# model = AutoModelForQuestionAnswering.from_pretrained("kirito546/QA-bert-multilingual")
# pre_answer=[]

# question=query
# context=GetSimilar(query,contexts)

# inputs = tokenizer(question, context, return_tensors="pt",padding=True, truncation=True)
# with torch.no_grad():
#     outputs = model(**inputs)

# answer_start_index = outputs.start_logits.argmax()
# answer_end_index = outputs.end_logits.argmax()

# predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
# print(tokenizer.decode(predict_answer_tokens))
   

