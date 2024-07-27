from flask import Flask, render_template,request
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import torch
from gg_search import *
from data import *
import pandas as pd



app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def QA():
    if request.method == 'POST':
        allTilte,dataset=Getdata()
        dataframe = pd.DataFrame(dataset)
        question = request.form['input_text']

        titlie= GetSimilar(question,allTilte)
        Ques_contexts= [i for i in GetContext(titlie)['question']]
        quetionInContext= GetSimilar("Viêm da tróc vảy là gì",Ques_contexts)
        fil_dataframe=dataframe.loc[dataframe['question']==quetionInContext].copy()
        contexts=[i for i in fil_dataframe['context']]

        context=contexts[0]
        model_select = request.form['model_select']

        if model_select == 'Bert-multilingual':
            tokenizer = AutoTokenizer.from_pretrained("kirito546/QA-bert-multilingual")
            model = AutoModelForQuestionAnswering.from_pretrained("kirito546/QA-bert-multilingual")
        elif model_select == 'PhoBERT':
            tokenizer = AutoTokenizer.from_pretrained("kirito546/QA-PhoBert")
            model = AutoModelForQuestionAnswering.from_pretrained("kirito546/QA-PhoBert")

        inputs = tokenizer(question, context, return_tensors="pt",padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer=tokenizer.decode(predict_answer_tokens)
        
        return render_template('normal.html',Title='QA',input_text=question,model_select=model_select,answer=answer)
    return render_template('normal.html',Title='QA',input_text=None, model_select=None,answer=None)

@app.route('/contextQA',methods=["GET","POST"])
def contextQA():
    if request.method == 'POST':
        question = request.form['question']
        model_select = request.form['model_select']
        context=request.form['context']

        if model_select == 'Bert-multilingual':
            tokenizer = AutoTokenizer.from_pretrained("kirito546/QA-bert-multilingual")
            model = AutoModelForQuestionAnswering.from_pretrained("kirito546/QA-bert-multilingual")
        elif model_select == 'PhoBERT':
            tokenizer = AutoTokenizer.from_pretrained("kirito546/QA-PhoBert")
            model = AutoModelForQuestionAnswering.from_pretrained("kirito546/QA-PhoBert")

        inputs = tokenizer(question, context, return_tensors="pt",padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer=tokenizer.decode(predict_answer_tokens)

        return render_template('context.html',Title='QA',question=question,model_select=model_select,context=context,answer=answer)
    return render_template('context.html',Title='QA',question=None,model_select=None,context=None,answer=None)

@app.route('/QA_online',methods=["GET","POST"])
def rag():
    if request.method == 'POST':
        question = request.form['input_text']
        model_select = request.form['model_select']
        query = question
        html = google_search(query)
        if html:
            contexts=[]
            search_results = parse_results(html)
            for result in search_results:
                soup = google_scrape(result['link'])
                if isinstance(soup, str):  # Check if there was an error
                    continue
                sections = split_content_by_h2(soup)
                for i in range(len(sections)):
                    sections[i]=clean_text(sections[i])
                for j in sections:
                    if len(j)<20:
                        sections.remove(j)
                if sections:
                    contexts.append(GetSimilar(query,sections))
        else:
            return render_template('rag.html',Title='QA',input_text=question,model_select=model_select,answer="Failed to retrieve search results")
        
        context=GetSimilar(query,contexts)


        if model_select == 'Bert-multilingual':
            tokenizer = AutoTokenizer.from_pretrained("kirito546/QA-bert-multilingual")
            model = AutoModelForQuestionAnswering.from_pretrained("kirito546/QA-bert-multilingual")
        elif model_select == 'PhoBERT':
            tokenizer = AutoTokenizer.from_pretrained("kirito546/QA-PhoBert")
            model = AutoModelForQuestionAnswering.from_pretrained("kirito546/QA-PhoBert")

        inputs = tokenizer(question, context, return_tensors="pt",padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer=tokenizer.decode(predict_answer_tokens)

        return render_template('rag.html',Title='QA',input_text=question,model_select=model_select,answer=answer)
    return render_template('rag.html',Title='online',input_text=None, model_select=None,answer=None)

if __name__ == '__main__':
    app.run(debug=True)