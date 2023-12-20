#!/usr/bin/env python
# coding: utf-8

##################################
# data extraction and cleaning
##################################


##################################
# gmail data
##################################

import mailbox
import csv


def getcharsets(msg):
    charsets = set({})
    for c in msg.get_charsets():
        if c is not None:
            charsets.update([c])
    return charsets

def handleerror(errmsg, emailmsg,cs):
    print()
    print(errmsg)
    print("This error occurred while decoding with ",cs," charset.")
    print("These charsets were found in the one email.",getcharsets(emailmsg))
    print("This is the subject:",emailmsg['subject'])
    print("This is the sender:",emailmsg['From'])


def getbodyfromemail(msg):
    body = None
    #Walk through the parts of the email to find the text body.    
    if msg.is_multipart():    
        for part in msg.walk():

            # If part is multipart, walk through the subparts.            
            if part.is_multipart(): 

                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        # Get the subpart payload (i.e the message body)
                        body = subpart.get_payload(decode=True) 
                        #charset = subpart.get_charset()

            # Part isn't multipart so get the email body
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
                #charset = part.get_charset()

    # If this isn't a multi-part message then get the payload (i.e the message body)
    elif msg.get_content_type() == 'text/plain':
        body = msg.get_payload(decode=True) 

   # No checking done to match the charset with the correct part. 
    for charset in getcharsets(msg):
        try:
            body = body.decode(charset)
        except UnicodeDecodeError:
            return ''
        except AttributeError:
            return ''
    return body


def filter_data_include(message, filters):
    for filter in filters:
        for (k, f) in filter.items():
            if k not in message or f not in message[k]:
                return False
    return True


def filter_data_exclude(message, filters):
    for filter in filters:
        for (k, f) in filter.items():
            if k in message and f in message[k]:
                return False
    return True



def write_csv(input_data="",output_data="",filters_include=None, filters_exclude=None):
    writer = csv.writer(open( output_data, 'w', newline='', encoding='utf-8'))
    headers=  ["Date","Subject","From","To","Body"]
    writer.writerow(headers) 
    for message in mailbox.mbox(input_data): 
        if filters_include and not filter_data_include(message, filters_include):
            continue
        if filters_exclude and not filter_data_exclude(message, filters_exclude):
            continue

        writer.writerow([
                message['Date'],
                message['Subject'],
                message['From'],
                message['To'],
                getbodyfromemail(message)] )

############################
####meetings
##########################

#load 

def read_meetings_data(input_data):
    import pandas as pd
    data = pd.read_csv(input_data)
    
    #drop if the file contains empty rows
    if data['Start'].isnull(). sum() > 0 :
        print("Removed empty rows")
        data=data.dropna(subset=['Start'])
        
    # convert date 
    from datetime import datetime
    dates = []
    
    for d in data['Start']:
        try:
            dt = datetime.strptime(d,"%d/%m/%Y")
            dates.append(datetime.strftime(dt, '%c'))
        except:
            dt = datetime.strptime(d,"%d/%m/%y")
            dates.append(datetime.strftime(dt, '%c'))

    data['Date_Converted'] = pd.Series(dates)

    data["Date_Converted_dt"] = pd.to_datetime(data.Date_Converted, utc=True)

    return data


#################################
# formality of meemtings 
########################################


def get_regular_meetings(data):
    import spacy
    from spacy.matcher import PhraseMatcher
    type_of_meet = data["Title"].to_list() #extract Title of meeting for the given month
    nlp = spacy.load('en_core_web_sm')
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

    # the list containing the pharses to be matched
    regularity_list = ["weekly", "week", "daily", "monthly", "status point"]      #complete the list if needed
    patterns = [nlp.make_doc(text) for text in regularity_list]
    matcher.add("Phrase Matching", None, *patterns)
    doc = nlp(str(type_of_meet))
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        #print(span.text)
        #print(match_id, string_id, start, end, span.text)
    results =  [doc[start:end].text for match_id, start, end in matches]
    regulars = len(results)
    return regulars

def calculate_formality_meetings(regulars, non_regular):
    regular_scheduled_weight = 2
    non_regular_weight = 1
    #calculate percentage, multiply by the weight, then normalize on a scale of 0 to 1:

    reg = regulars/(regulars + non_regular)*10
    non_reg = non_regular/(regulars + non_regular)*10

    total = (reg * regular_scheduled_weight) + (non_reg * non_regular_weight)

    return total

def get_formality_meetings(input_data):
    import calendar
    data = read_meetings_data(input_data)
    available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
    dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]
    formality_meetings = dict()
    for i, df in zip(available_months, dfs):
        try:
            index = df.index
            no_of_meetings = len(index)
            regulars = get_regular_meetings(df)
            non_regular = no_of_meetings - regulars
            total = calculate_formality_meetings(regulars, non_regular)

            formality_meetings.update({calendar.month_name[i] : total})
        except KeyError:
            print(f"key error for month {i}")
            continue
    return formality_meetings

def normalize_with_rolling_max(months, values):
    max_v = max(values)
    normalized = [(round(value/max_v, 2)) for value in values]
    d = {k:v for k, v in zip(months,normalized)}
    return d

def get_n_formality_meetings(formality_meetings):
    months = list(formality_meetings.keys())
    values = list(formality_meetings.values())
    n_formality_meetings = normalize_with_rolling_max(months, values)
    return n_formality_meetings

##################################################
############################################
## Language sophistication score 
#######################################
# function to remove prefixes in subjects

def remove_prefixes_in_sbj(subject):
    import re
    p = re.compile( '([\[\(] *)?(RE?S?|FYI|RIF|I|FS|VB|RV|ENC|ODP|PD|YNT|ILT|SV|VS|VL|AW|WG|ΑΠ|ΣΧΕΤ|ΠΡΘ|תגובה|הועבר|主题|转发|FWD?) *([-:;)\]][ :;\])-]*|$)|\]+ *$', re.IGNORECASE)
    return p.sub( '', subject).strip()

def clean(text):
    import re
    import string
    from six.moves.html_parser import HTMLParser
    import contractions
    import html as html
    h = HTMLParser()
    target = string.printable + "öäüÖÄÜ"

    
    sentences = []
    for subject in text:
        if type(subject) is not str:
            continue
        new1 = re.sub("(https?[^\s]+)", " ", subject) #removing urls
        #new2 = h.unescape(new1) #converting other HTML entities to recongisable characters
        new2 = html.unescape(new1)

        new3 = re.sub("^(re:|fw:|fwd:|Re: Fw:|Re:|Fwd:|Fw:|RE: FW:|FW:|RE: Rdv|Rdv|RE:|FW:|AW:|WG:)", " ",new2)
        new4 = re.sub("(re:|fw:|fwd:|Re: Fw:|Re:|Fwd:|Fw:|RE: FW:|FW:|RE: Rdv|Rdv|RE:|FW:|AW:|R:|WG:)", " ",new3)
        new5 = contractions.fix(new4)
        sentences.append(new4)
    return sentences

def read_messages_data(input_data=None):
    from datetime import datetime
    import pandas as pd
    data = pd.read_csv(input_data)
    dates = []
    for d in data['Date']:
        try:
            d = d.replace(' (CET)', '')
            d = d.replace(' (UTC)', '')
            d = d.replace(' (PST)', '')
            d = d.replace(' (CEST)', '')
            dates.append(datetime.strptime(d, "%a, %d %b %Y %H:%M:%S %z"))
        except:
            print(d)
            dates.append(d)

    data['Date_Converted'] = pd.Series(dates)
    #data = data.sort_values('Date_Converted', ascending=True) ###sort all data by Date_converted values

    # apply removing replies prefixes in the subjects:
    #data["Subject"] = data["Subject"].astype('str').apply(remove_prefixes_in_sbj)
    
    # convert Date_Converted to datetime in order to sort df by months
    data["Date_Converted_dt"] = pd.to_datetime(data.Date_Converted, utc=True)

    return data
    
def percentage(part, whole):
  percentage = 100 * float(part)/float(whole)
  return round(percentage)

def get_F_sophistication(text):
	import nltk
	nltk.download('averaged_perceptron_tagger', quiet= True)
	nltk.download('punkt', quiet= True)
	from collections import Counter

	words = nltk.word_tokenize(str(text))
	tags = nltk.pos_tag(words)
	all_tags = len(tags)
	counts = Counter( tag for word,  tag in tags)
    #print(counts)
    #print(all_tags)

	nouns = counts["NNP"] + counts["NN"] + counts["NNS"]+ counts["NNPS"]
	noun_freq = percentage(nouns, all_tags)
    #print(noun_freq)

	adjectives = counts["JJ"] + counts["JJR"]+ counts["JJS"]
	adjective_freq = percentage(adjectives, all_tags)
    #print(adjective_freq)

	verbs = counts["VB"] + counts["VBD"] + counts["VBZ"] + counts["VBG"] + counts["VBP"] + counts["VBN"] + counts["MD"]
	verb_freq = percentage(verbs, all_tags)
    #print(verb_freq)

	pronouns = counts["PRP"] + counts["PRP$"] + counts["WP"]
	pronoun_freq = percentage(pronouns, all_tags)
    #print(pronoun_freq)

	articles = counts["DT"] + counts["WDT"]
	article_freq = percentage(articles, all_tags)
    #print(article_freq)

	prepositions = counts["IN"]
	preposition_freq = percentage(prepositions, all_tags)
    #print(preposition_freq)

	adverbs = counts["RB"] + counts["RBR"]+counts["RBS"] + counts["WRB"]
	adverb_freq = percentage(adverbs, all_tags)
    #print(adverb_freq)

	interjections = counts["UH"]
	interjection_freq = percentage(interjections, all_tags)
    #print(interjection_freq)

    #F will then vary between 0 and 100%.
    #The more formal/sophisticated the language is, the higher the value of F is expected to be.
	F = ((noun_freq + adjective_freq + preposition_freq + article_freq) - (pronoun_freq + verb_freq + adverb_freq + interjection_freq) + 100)/2
    
	return F/100

def get_members(data):
    sender = data['From'].unique()
    #remove Error 
    sender = sender[sender !="#ERROR!"]
    return sender


def get_written_formality(input_data):
    import calendar
    data = read_messages_data(input_data)
    print("data read fin")
    #available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
	#dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]

    available_months = sorted(list(data.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [data[data.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]
    #calculation by month
    monthly_written_formality = dict()

    for i, df in zip(available_months, dfs):
        #loop by member
        sender = get_members(df)
        within_month = dict()
        for sdr in sender : 
            dfm = df[df['From']==sdr]
            #print(i, sdr, len(dfm))
            if len(dfm)==0 :
                next
            text =dfm["Body"].to_list()
            clean_values = clean(text)
            within_month.update({sdr : get_F_sophistication(clean_values)})
        #monthly_written_formality.update({calendar.month_name[i] : within_month})
        monthly_written_formality.update({str(i) : within_month})


    return monthly_written_formality

def get_n_written_formality(written_formality):
    import statistics
    n_written_formality = dict()
    for month in written_formality.keys() : 
        month_formality = written_formality[month]
        mean_value = statistics.mean(month_formality.values())
        n_written_formality.update({month : mean_value})
    return(n_written_formality)

##################################
#work formalisation 
##################################
def remove_duplicates(sents):
  unique_sents = []
  for sent in sents:
    if sent not in unique_sents:
      unique_sents.append(sent)
  return unique_sents

def clean_work_formalization(texts):
    import re
    import string
    from six.moves.html_parser import HTMLParser
    import contractions
    import html as html

    h = HTMLParser()
    target = string.printable + "öäüÖÄÜ"

    
    sentences = []
    for v in texts:
        if type(v) is not str:
            continue
        doc = re.sub ("\n", " ", v)
        doc = re.sub ("\r", " ",doc)
        doc = re.sub("http\S+", " ", doc) #removing urls
        doc = re.sub("[`~!@#$%^&*()_|+\-=?;:'<>\{\}\[\]\\\/]", ' ', doc)
        #doc = h.unescape(doc) #converting other HTML entities to recongisable characters
        doc = html.unescape(doc)
        doc = re.sub("(\s|\t){2,}", " ", doc) #removing unnecessary spaces
        doc = re.sub("\S*@\S*\s?", " ", doc) #removing emails
        doc = re.sub("@[^\d]", " ", doc) #removing phone numbers
        sentences.append(doc.strip()) #removing leading and trailing spaces and adding clean string to the list
    return sentences

def preprocess(doc):
    from re import sub
    from gensim.utils import simple_preprocess

    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'\n', '', doc)
    doc = sub(r'http\S+', '', doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, deacc=False, min_len=2, max_len=15)]

def get_work_formalisation(input_data):
    from re import sub
    from gensim.utils import simple_preprocess
    import numpy as np
    import gensim.downloader as api
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel
    from gensim.similarities import WordEmbeddingSimilarityIndex
    from gensim.similarities import SparseTermSimilarityMatrix
    from gensim.similarities import SoftCosineSimilarity
    # Load the model: this is a big file, can take a while to download and open
    glove = api.load("glove-wiki-gigaword-50")    
    similarity_index = WordEmbeddingSimilarityIndex(glove)
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import stopwords
 #   nltk.download('stopwords', quiet=True)
 #   nltk.download('punkt', quiet = True)
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer 
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics.pairwise import euclidean_distances
    from langdetect import detect
    import language_tool_python
    tool = language_tool_python.LanguageTool('en-US')
    from googletrans import Translator
    translator = Translator()
    import calendar

    ###dictionary with contextual phrases###
    d = dict()	
    d['organizational'] = ["Work organisation thus refers to how work is planned, organised and managed within companies and to choices on a range of aspects such as work processes, job design, responsibilities, task allocation, work scheduling, work pace, rules and procedures, and decision-making processes."]
    d['revision'] = ["revision", "revise", "redo", "revising", "re-", "revised", "redone", "redraft"]

    data = read_messages_data(input_data)
    data.Body = data.Body.str.replace("\n","")
    #data.loc[data.Body==" ", ["Body"]]="empty"
    #data['Translated'] = data['Body'].apply(lambda x: translator.translate(x, dest='en').text)
    #Translate text to English
    data['Translated'] = ""
    for index, row in data.iterrows():
        if pd.isna(row.loc['Body']):
            continue
        if row.loc['Body']=='https://edgeryders.eu/c/earthos/playful-futures/427':
            continue
        if row.loc['Body'].replace("\n","").strip() !="" :
            data.loc[index,'Translated']=translator.translate(row.loc['Body'],dest='en').text
    data.loc[data.Body=="empty", ["Body"]]=""
    data.loc[data.Body=="empty", ["Translated"]]=""


    #available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
	#dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]
    available_months = sorted(list(data.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [data[data.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]

    organization = dict()
    revision = dict()
    for i, df in zip(available_months, dfs):
        text = df["Translated"].to_list()
        subjects = remove_duplicates(text)
        clean_sents = clean_work_formalization(subjects)
        full_text = '.'.join(str(x) for x in clean_sents)
        splitted_sentences = nltk.sent_tokenize(full_text)
        ###creating new table
        pd.set_option('display.max_colwidth', 0)
        pd.set_option('display.max_columns', 0)
        documents_df=pd.DataFrame(splitted_sentences,columns=['splitted_sentences'])
        df_new = documents_df[documents_df['splitted_sentences'].notnull()]
        text_messag = df_new['splitted_sentences'].to_list()
        new_subjects = remove_duplicates(text_messag)

        samples = {}
        for (name, v) in d.items():
            query_string = str(v)

            documents = new_subjects
                            
            # Preprocess the documents, including the query string
            corpus = [preprocess(document) for document in documents]
            query = preprocess(query_string)
            raw_corpus = len(corpus)
            #print(raw_corpus)

            # Build the term dictionary, TF-idf model
            dictionary = Dictionary(corpus+[query])
            tfidf = TfidfModel(dictionary=dictionary)

            # Create the term similarity matrix.  
            similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)


            # Compute Soft Cosine Measure between the query and the documents.
            query_tf = tfidf[dictionary.doc2bow(query)]

            index = SoftCosineSimilarity(
                        tfidf[[dictionary.doc2bow(document) for document in corpus]],
                        similarity_matrix)

            doc_similarity_scores = index[query_tf]

            # Output the sorted similarity scores and documents
            sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
            samples[name] = 0
            #print(name)  
            for idx in sorted_indexes:
                if doc_similarity_scores[idx] > 0.3:
                    samples[name] = samples[name] + 1
        len_org = samples.get("organizational")
        len_rev = samples.get("revision")
        org_month_value = len_org/raw_corpus
        rev_month_value = len_rev/raw_corpus

        #organization.update({calendar.month_name[i] : org_month_value})
        organization.update({str(i) : org_month_value})
        #revision.update({calendar.month_name[i] : rev_month_value})
        revision.update({str(i) : rev_month_value})
    return organization, revision


#############################################
### Visualization 
########################################
def plt_formality_meeting(n_formality_meetings, outdir) : 
    #Changes in formality of meetings Sub-indicator visualization, data is already normalized in its dedidated notebook
    import plotly.graph_objects as go
    # Add data
    month = list(n_formality_meetings.keys())
    scores = list(n_formality_meetings.values())
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Changes in formality of meetings',
                            line=dict(color='orange', width=4)))
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
    # Edit the layout
    fig.update_layout(#title='<b>Changes in formality of meetingsSub-indicator</b><br>\
                    xaxis_title='Month',
                    yaxis_title='Score')
    fig.update_layout(yaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale (min 0, max 1)<i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="grey")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.write_image(outdir  +"\\formality_meetings.png")
    #fig.show()

def plt_written_formality(n_written_formality, outdir):
    #Changes in formality of written style Sub-indicator visualization, data is already normalized in its dedidated notebook
    import plotly.graph_objects as go
    # Add data
    month = list(n_written_formality.keys())
    scores = list(n_written_formality.values())
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Changes in formality of written styles',
                            line=dict(color='green', width=4)))
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
    # Edit the layout
    fig.update_layout(#title='<b>Changes in formality of meetingsSub-indicator</b><br>\
                    xaxis_title='Month',
                    yaxis_title='Score')
    fig.update_layout(yaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale (min 0, max 1)<i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="grey")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.write_image(outdir  +"\\written_formality.png")
    #fig.show()


def plt_organization(organization, outdir):
    #Changes in work formalisation Sub-indicator visualization, no need to apply normalization, so we use raw data from dedicated notebook
    import plotly.graph_objects as go
    # Add data
    month = list(organization.keys())
    scores = list(organization.values())
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Changes in work formalisation',
                            line=dict(color='red', width=4)))
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
    # Edit the layout
    fig.update_layout(#title='<b>Changes in work formalisation Sub-indicator</b><br>\
                    xaxis_title='Month',
                    yaxis_title='Score')
    fig.update_layout(yaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale (min 0, max 1)<i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="grey")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.write_image(outdir  +"\\work_organization.png")
    #fig.show()

def plt_revision(revision, outdir):
    #Changes in work formalisation Sub-indicator visualization, no need to apply normalization, so we use raw data from dedicated notebook
    import plotly.graph_objects as go
    # Add data
    month = list(revision.keys())
    scores = list(revision.values())
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Changes in work formalisation',
                            line=dict(color='red', width=4)))
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
    # Edit the layout
    fig.update_layout(#title='<b>Changes in work formalisation Sub-indicator</b><br>\
                    xaxis_title='Month',
                    yaxis_title='Score')
    fig.update_layout(yaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale (min 0, max 1)<i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="grey")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.write_image(outdir  +"\\work_revision.png")
    #fig.show()

def plt_formality_macro(revision,organization, n_written_formality,n_formality_meetings, outdir):
    # all the subindicators on one figure
    import plotly.graph_objects as go
    # Add data
    month = list(revision.keys())
    revision = list(revision.values())
    organization = list(organization.values())
    language = list(n_written_formality.values())
    meetings = list(n_formality_meetings.values())

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=month, y=revision, name='Moments of work revision',
                            line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=month, y=organization, name = 'Changes in work formalisation',
                            line=dict(color='blue', width=4)))
    fig.add_trace(go.Scatter(x=month, y=language, name='Changes in formality of written style',
                            line=dict(color='orange', width=4,
                                dash='dash') # dash options include 'dash', 'dot', and 'dashdot'
    ))
    fig.add_trace(go.Scatter(x=month, y=meetings, name='Changes in formality of meetings',
                            line = dict(color='green', width=4, dash='dash')))
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
    # Edit the layout
    fig.update_layout(
        #title='<b>Change of Formality Sub-indicators <br>Scale (min 0, max 1)</b>',
                    xaxis_title='Month',
                    yaxis_title='Score')

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale (min 0, max 1)<i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="grey")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.write_image(outdir  +"\\formality_macro.png")
    #fig.show()