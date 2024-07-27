import json
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def Getdata():
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Chuyển đổi dữ liệu thành format của Dataset
    contexts = []
    questions = []
    answers = []
    ids = []
    titles=[]
    allTilte=[]
    for i in data['data']:
        title=i['title']
        allTilte.append(title)
        for j in i['paragraphs']:
            context = j['context']
            for qa in j['qas']:
                question = qa['question']
                id_ = qa['id']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                    ids.append(id_)
                    titles.append(title)


    dataset = Dataset.from_dict({
        'id': ids,
        'context': contexts,
        'question': questions,
        'answers': answers,
        'title':titles
    })

    return allTilte,dataset

def GetContext(input_title):
    allTilte,dataset=Getdata()

    df = pd.DataFrame(dataset)
    for i in range(len(df)):
        df['title'][i]=df['title'][i].lower()
    filtered_df = df.loc[df['title']== input_title.lower()].copy()
    return filtered_df


def GetSimilar(question,contexts):
    documents = contexts + [question]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    question_vector = tfidf_matrix[-1]
    context_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(question_vector, context_vectors)
    best_match_index = similarities.argmax()
    context = contexts[best_match_index]
    return context


# allTilte,dataset=Getdata()
# dataframe = pd.DataFrame(dataset)
# titlie= GetSimilar("Viêm da tróc vảy là gì",allTilte)
# Qcontexts= [i for i in GetContext(titlie)['question']]
# a= GetSimilar("Viêm da tróc vảy là gì",Qcontexts)

# fil_dataframe=dataframe.loc[dataframe['question']==a].copy()
# b=[i for i in fil_dataframe['context']]
# print(b[0])

# for i in allTilte:
#     print(i)
# data=GetContext("bạch tạng")

# contexts=[]
# for i in data['context']:
#   if i not in contexts:
#     contexts.append(i)
# bert_context_vectors = bert_embed_data(contexts)

# print(GetSimilarBert('đối phó với bệnh bạch tạng',contexts))