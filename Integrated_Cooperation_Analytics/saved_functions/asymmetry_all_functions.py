#!/usr/bin/env python -i
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



##################################################
############################################
## message data
#######################################

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

    data["Date_Converted_dt"] = pd.to_datetime(data.Date_Converted, utc=True)

    return data
    

# get all the emails in a merged string format
def get_all_emails_str(data):
    from math import nan, isnan
    import re
    all_from = list(data["From"].values)
        
    mail_without_nan = [x for x in all_from if str(x) != 'nan']
    all_values = " ".join(mail_without_nan)
    match_values = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', all_values)
    all_mails_str = " ".join(list(set(match_values)))
    return all_mails_str.split()

def get_users_sent_messages_as_str(data, list_of_emails):
    import pandas as pd
    list_of_frames = []
    for email in list_of_emails:
        sent_user_data = data[data.apply(lambda x: email in x.From, axis=1)]
        list_of_frames.append(sent_user_data)
    data = pd.concat(list_of_frames)
    # we need translated version of emails bodies
    data = data.Translated
    data = data.to_numpy() 
    # combine text data of messages sent by participants into string
    text = " ".join(data)
    text = " ".join(text.splitlines())
    return text

# create dataframe that contains only sent messages of Pilot 7 senders emails
def get_users_sent_messages_df(data, list_of_emails):
    import pandas as pd
    list_of_frames = []
    for email in list_of_emails:
        sent_user_data = data[data.apply(lambda x: email in x.From, axis=1)]
        list_of_frames.append(sent_user_data)
    return pd.concat(list_of_frames)


# function for removing emojies
def remove_emojis(text):
    import re
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) # no emoji

def preprocess_text(text, lang="en"):
    import re
    import contractions
    text = text.replace(" the ", " ")
    text = re.sub(r"/([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/gi", "", text)# remove email adresses
    #Extend contractions
    text = contractions.fix(text)
    text = remove_emojis(text)
    text = re.sub(r"\[.*?\]", " ", text) #remove links to images and everything in square brackets
    text = re.sub("\\r|\\n", " ", text) #removing newlines and carriage return
    text = re.sub("(https?[^\s]+)", " ", text) #removing urls
    text = re.sub("[^a-zA-Z]+", " ", text)# remove non-alphabetic characters
    text = re.sub(r'\s+', " ", text)# remove more than 1 space
    text = text.strip() #removing leading and trailing spaces and adding clean string to the list
    return text

# extract candidate terms

def extract_terms(clean_test):  
    import tm2tb
    from tm2tb import TermExtractor
    extractor = TermExtractor(clean_test)  # Instantiate extractor with sentence
  
    terms = extractor.extract_terms(span_range=(1,2), incl_pos= ['NOUN'], freq_min = 1, excl_pos=['PROPN','DT',"VERB" 'RB', "VBD", "NNP", "NNPS"])  # Extract terms, selecting frequency, the terms length and the parts-of-speech tags (terms) to include or to exclude
    new_terms = terms["term"].to_numpy()
    return list(new_terms)


def filter_terms(new_terms):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    #join extracted term phrases into 1 string
    terms_joined_str = " ".join(list(new_terms))
    terms_separate_words = terms_joined_str.split(" ")
    #leave only unique words
    terms_to_save = list(set(terms_separate_words))
    #check if separate words from terms phrases are not in stop words
    filtered_chunks = list(set([w for w in terms_to_save if not w.lower() in stopwords.words("english") and len(w)>3]))
    # create a mapping of word to its lemmatized form for checking in the dictionary of common words
    terms_map = {w:wnl.lemmatize(w) for w in filtered_chunks}
    # load dictionary of common words
    #load the dictionary of common words
    diction = open('daily_words_eng.txt', "r")
    daily_words = diction.read()
    terms_to_use = []
    for raw, lemm in terms_map.items():
        # check if lemmatized form encounters in common words, if not, save its raw extracted form for further analysis
        if lemm not in daily_words:
            terms_to_use.append(raw)
                
    # get all the email adresses as a merged str for easy checking if dected terms are not part of email addresses
    #all_mails_str = get_all_emails_str(data)
    # consider as terms words which length is more than 3 characters
    terms_to_use = [t for t in terms_to_use if len(t)>3]
#     print(len(terms_to_use))

    return terms_to_use

def get_difference(list1, list2):
    return list(set(list2).difference(list1))

def calculate_term_score(month_df, month_terms, num_of_participants):
    import re
    term_to_score = {}
#     num_of_participants = 2
    for term in month_terms:
        for index, row in month_df.iterrows():
                # check messages with no subject:
                if term.lower() in str(row["Translated"]).lower():
                    if term not in term_to_score:
                        term_to_score[term]=[(row["Date_Converted_dt"], row["From"])]
                    else:
                        term_to_score[term].append(((row["Date_Converted_dt"], row["From"])))

    
    term_to_users_frequency = {}
    for key in list(term_to_score.keys()):
        #Find first occurrence of term by date and detect person who introdused it
        first_sender = sorted(list(term_to_score[key]))[0][1]
        # get all the email addresses who reproduced a term
        unique_emails = set([el[1] for el in list(term_to_score[key])])
        emails = [el[1] for el in list(term_to_score[key])]
        # count how many times each sender reproduced the term
        email_to_term_count = {em:emails.count(em) for em in unique_emails}
        # calculate total score of the term as total count of the number of reproductions by all the users
        score = sum({em:emails.count(em) for em in unique_emails}.values())
        max_score = num_of_participants + 1
        # when score is equal to max value or bigger: 
        if score >= max_score:
            # check if the first sender reproduced the term + another persons used it:
            if email_to_term_count[first_sender]>=2:
                if len(unique_emails) == num_of_participants:
                    score = max_score
                else:
                    score = len(unique_emails) + 1
            # when another person used term more than once, but the first user did not reproduce:
            else:
                score = len(unique_emails)
        # when score is lower than the max score
        else:
            if email_to_term_count[first_sender]>=2 and len(unique_emails)>1:
                score = len(unique_emails)+1
            
            else:
                # if only first sener reproduced their term
                score = len(unique_emails)
        term_to_score[key]=(score, re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', first_sender)[0])
    return term_to_score

def user_to_introduced_terms(term_to_score_to_sender):
    user_to_introduced_terms = {}
    for k,v in term_to_score_to_sender.items():
        
        if v[1] not in user_to_introduced_terms:
        # if 2 separate users used term or max value when first sender twice and another person once used it
            user_to_introduced_terms[v[1]] = [(k, v[0])]
        else:
            user_to_introduced_terms[v[1]].append((k, v[0]))

    user_to_introduced_terms = {k: [{el[0]:el[1]} for el in v] for k,v in user_to_introduced_terms.items()}
    user_to_introduced_terms = {key: {k: v for d in value for k, v in d.items()} for key, value in user_to_introduced_terms.items()}
    return user_to_introduced_terms

def format_for_pieplot_by_user(month_to_final_result):
    new_dic = {}
    for k, v  in month_to_final_result.items():

        new_dic[k] = {k:sum(list(v.values())) for k, v in v.items()}
    return new_dic

def balance(seq, k=None):
    from collections import Counter
    from numpy import log
    
    n = sum(seq.values())
    classes = [(clas,float(count)) for clas,count in Counter(seq).items()]
    if not k:
        k = len(classes)
    H = -sum([(count/n) * log((count/n)) for clas,count in classes if count!=0])
    return H/log(k)



def get_translated_data(input_data):
    data = read_messages_data(input_data)
    print("data read fin")
    
    ##translate
    from googletrans import Translator
    import pandas as pd
    translator = Translator()

    #Translate text to English
    data['Translated'] = ""
    for index, row in data.iterrows():
        if pd.isna(row.loc['Body']):
            continue
        if row.loc['Body']=='https://edgeryders.eu/c/earthos/playful-futures/427':
            continue
        if row.loc['Body'].replace("\n","").strip() !="" :
            data.loc[index,'Translated']=translator.translate(row.loc['Body'],dest='en').text
    #data['Translated'] = data['Body'].apply(lambda x: translator.translate(x, dest='en').text)
    
    return data 


def get_month_to_result(data) : 
    list_senders = get_all_emails_str(data)
    df_sent_mess = get_users_sent_messages_df(data, list_senders)
    # available_months = sorted(list(df_sent_mess.Date_Converted_dt.dt.month.unique()))
    # dfs = [df_sent_mess[df_sent_mess.Date_Converted_dt.dt.month == month] for month in available_months]

    available_months = sorted(list(df_sent_mess.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [df_sent_mess[df_sent_mess.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]

    #calculation by month
    month_to_result = {}
    for i, df in zip(available_months, dfs):
        all_data = get_users_sent_messages_as_str(df, list_senders)
        clean_text = preprocess_text(all_data)
        try:
            terms_extr = extract_terms(clean_text)
            print("\nAll therms phrases: ", terms_extr)
        except ValueError:
            print("No terms detected")
            continue
        terms_to_check = filter_terms(terms_extr)
        print("\nTherms after filtering for common words: ", terms_to_check)

        month_to_result.update({str(i) : terms_to_check})
    return month_to_result

# def get_month_to_result(data) : 
#     import calendar

#     list_senders = get_all_emails_str(data)
#     df_sent_mess = get_users_sent_messages_df(data, list_senders)
#     available_months = sorted(list(df_sent_mess.Date_Converted_dt.dt.month.unique()))


#     dfs = [df_sent_mess[df_sent_mess.Date_Converted_dt.dt.month == month] for month in available_months]

#     #calculation by month
#     month_to_result = {}
#     for i, df in zip(available_months, dfs):
#         all_data = get_users_sent_messages_as_str(df, list_senders)
#         clean_text = preprocess_text(all_data)
#         try:
#             terms_extr = extract_terms(clean_text)
#             #print("\nAll therms phrases: ", terms_extr)
#         except ValueError:
#             #print("No terms detected")
#             continue
#         terms_to_check = filter_terms(terms_extr)
#         #print("\nTherms after filtering for common words: ", terms_to_check)

#         month_to_result.update({calendar.month_name[i] : terms_to_check})
#     return month_to_result

def get_difference_with_previous_terms(data, month_to_result, month):
    import calendar

    list_senders = get_all_emails_str(data)
    df_sent_mess = get_users_sent_messages_df(data, list_senders)
    # available_months = sorted(list(df_sent_mess.Date_Converted_dt.dt.month.unique()))
    # dfs = [df_sent_mess[df_sent_mess.Date_Converted_dt.dt.month == month] for month in available_months]

    available_months = sorted(list(df_sent_mess.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [df_sent_mess[df_sent_mess.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]

    i=month
    if calendar.month_name[i] in list(month_to_result.keys()):
        if  calendar.month_name[i] != list(month_to_result.keys())[0]:
            prev_list =list()
            k =0
            while available_months[k] != i :
                month_md = month_to_result[calendar.month_name[available_months[k]]]
                #print(month_md)
                #month_list= [item for month_md]
                prev_list.append(month_md)
                k +=1
            previous_list = [item for sublist in prev_list for item in sublist]

            return get_difference(previous_list,month_to_result[calendar.month_name[i]])



# def get_month_to_final_result(data, month_to_result):
#     import calendar

#     list_senders = get_all_emails_str(data)
#     df_sent_mess = get_users_sent_messages_df(data, list_senders)
#     available_months = sorted(list(df_sent_mess.Date_Converted_dt.dt.month.unique()))
#     dfs = [df_sent_mess[df_sent_mess.Date_Converted_dt.dt.month == month] for month in available_months]
    
    
#     for i, df in zip(available_months, dfs):
#         if calendar.month_name[i] in list(month_to_result.keys()):
#             if  calendar.month_name[i] != list(month_to_result.keys())[0]:
#                 monthly_terms = get_difference_with_previous_terms(data,month_to_result, i)
#                 result_monthly = calculate_term_score(df, monthly_terms, num_of_participants=len(list_senders))
#                 df_to_terms_extracted = zip([df],[monthly_terms])
#                 month_to_final_result = {}
#                 months = [calendar.month_name[i]]
#                 count = 0
#                 for df, terms in df_to_terms_extracted:
#                     result = calculate_term_score(df, terms, num_of_participants=len(list_senders))
#                     final_result = user_to_introduced_terms(result)
#                     month_to_final_result[months[count]] = final_result
#                     count+=1
#     return month_to_final_result

def get_month_to_final_result_from_store_data(data, stored_data, month_to_result):
    from datetime import date, timedelta, datetime
    import pandas as pd
    list_senders = get_all_emails_str(data)
    df_sent_mess = get_users_sent_messages_df(data, list_senders)
    month = sorted(list(df_sent_mess.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    month_d = datetime.strptime(month[0], "%Y-%m")
    current_month = datetime.strftime(month_d, "%Y-%m")
    previous_month =  datetime.strftime(month_d - timedelta(days = 15),"%Y-%m")
    
    if pd.isna(stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",previous_month].item()) : 
        monthly_terms = list(month_to_result.values())[0]
    else:
        previous_list = stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",previous_month].str.split(",")
        monthly_terms = get_difference(list(previous_list)[0],list(month_to_result.values())[0])

    result_monthly = calculate_term_score(data, monthly_terms, num_of_participants=len(list_senders))
    df_to_terms_extracted = zip([data],[monthly_terms])
    month_to_final_result = {}
    final_result = user_to_introduced_terms(result_monthly)
    month_to_final_result[current_month] = final_result

    return month_to_final_result, monthly_terms

def get_n_knowledge_asymmetry(data, month_to_final_result):
    list_senders = get_all_emails_str(data)
    result_for_pieplot = format_for_pieplot_by_user(month_to_final_result)
    # calculation of the knowledge distribution balance score (the lower is score, the higher is asymmetry)
    months = list(result_for_pieplot.keys())
    shannon_results = {}
    for month in months:
        month_score_result = result_for_pieplot[month]
        shannon_results[month]=round(balance(month_score_result, k=len(list_senders)), 2)

    return shannon_results
#####

#######################################
## visualization
########################################

def get_terms_result_by_month_for_plot(month_to_final_result):
    terms_result_by_month_for_plot = {}
    months = list(month_to_final_result.keys())
    for month in months:
        list_of_dicts = [el for el in list(month_to_final_result[month].values())]
        dic = {k: v for d in list_of_dicts for k, v in d.items()}
        terms_result_by_month_for_plot[month] = dic
    return terms_result_by_month_for_plot


def generate_wordcloud_fig_month(result, selected_month = None, outdir =None):
    import plotly.graph_objects as go
    from wordcloud import WordCloud
    result_m = result[selected_month]
    
    wc = WordCloud(background_color="black").generate_from_frequencies(result_m)

    fig = go.Figure()
    fig.add_trace(go.Image(z=wc))
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        hovermode=False,
    )
    fig.update_layout(width = 1000) # size of the figure
    fig.update_layout(title=f'Result for <b>{selected_month}</b> <br>')
    #fig.show()
    fig.write_image(outdir  +"\\asymmetry_wordcloud_"+selected_month+".png")

    return fig

def generate_wordcloud_fig_member(result, selected_month = None, participant=None, outdir =None):
    import plotly.graph_objects as go
    from wordcloud import WordCloud

    result_m = result[selected_month]
    if participant:
        result_m = result_m[participant]
    
    wc = WordCloud(background_color="black").generate_from_frequencies(result_m)

    fig = go.Figure()
    fig.add_trace(go.Image(z=wc))
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        hovermode=False,
    )
    fig.update_layout(width = 1000) # size of the figure

    participant_name = "No_member_selected"
    if participant:
      fig.update_layout(title=f'Participant <b>{participant}</b> introduced the terms in <b>{selected_month}</b> :<br>')
      fig.update_layout(title={'font': {'size': 20}})
      participant_name = participant.split('@')[0]
    #fig.show()
    fig.write_image(outdir  +"\\asymmetry_wordcloud_"+selected_month+"_for_"+participant_name+".png")
    return fig

def get_wordcloud_fig_by_month(result, outdir=None):
    result = get_terms_result_by_month_for_plot(result)
    months = list(result.keys())
    for month in months:
        generate_wordcloud_fig_month(result,month, outdir=outdir)

def get_wordcloud_fig_by_members(result, outdir =None):
    months = list(result.keys())
    for month in months:
        montly_sender = list(result[month].keys())
        for s in montly_sender :
            generate_wordcloud_fig_member(result,month, s, outdir = outdir)     

def get_pie_plot_by_month(month_to_final_result,outdir):
    import matplotlib.pyplot as plt
    result_for_pieplot = format_for_pieplot_by_user(month_to_final_result)
    months = list(result_for_pieplot.keys())

    for month in months:
        print("")
        print(month + " :")
        month_score_result = result_for_pieplot[month]
        # Get the Keys and store them in a list
        first_senders = list(month_score_result.keys())
        # Get the Values and store them in a list
        total_scores = list(month_score_result.values())
    # import matplotlib.pyplot as plt
        plt.figure(figsize=(5,5))
        plt.style.use('seaborn-pastel')
        plt.pie(total_scores, labels=first_senders, autopct='%1.0f%%', shadow = True)
        plt.title(f"{month}: Knowledge distribution by project participants\n")
        #plt.show()
        plt.savefig(outdir  +"\\asymmetry_pieplot_"+month+".png")


###########################################
###Blog 
##########################################
def preprocess(doc):
    from re import sub
    from gensim.utils import simple_preprocess
    stopwords = ['the', 'and', 'are', 'a', "/"]

    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]

def preprocess_sw(doc):
    from re import sub
    from gensim.utils import simple_preprocess
    from nltk.corpus import stopwords
    stopwords =stopwords.words('english')
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]

#Normalization has to be applied now - for this we use Shannon entropy formula (0 - symmetric, 1 - assymetric)
def balance_blog_counts(seq, k=None):
    from collections import Counter
    from numpy import log
    
    n = sum(seq.values())
    classes = [(clas,float(count)) for clas,count in Counter(seq).items()]
    if not k:
        k = len(classes)
    H = -sum([(count/n) * log((count/n)) for clas,count in classes if count!=0])
    return H/log(k)


def get_ibm_category_blog(mixed_url =None, api_key=None,filename=None):

    # Load the model: this is a big file, can take a while to download and open
    import json
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, ConceptsOptions, CategoriesOptions, EntitiesOptions

    authenticator = IAMAuthenticator(api_key)  #insert here your NLU API key
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )

    natural_language_understanding.set_service_url('https://api.eu-de.natural-language-understanding.watson.cloud.ibm.com/instances/8d485964-c1f1-4001-b9ba-09f99451baca')

    response = natural_language_understanding.analyze(
    url = mixed_url,  #insert here URL to Blogs (better option will be to generate one mixted URL for all articles)
    features=Features(categories=CategoriesOptions(limit=1))).get_result()

    with open(filename + "_categories_blog.json", "w") as out_file:
        json.dump(response, out_file, indent=2)

    with open(filename + "_categories_blog.json") as json_file:
        data = json.load(json_file)
        for p in data['categories']:
            category = p['label']
            ibm_category = category.replace("/", " ")
    return ibm_category

def get_ibm_category_blog_from_json(filename=None):

    # Load the model: this is a big file, can take a while to download and open
    import json

    with open(filename + "_categories_blog.json") as json_file:
        data = json.load(json_file)
        for p in data['categories']:
            category = p['label']
            ibm_category = category.replace("/", " ")
    return ibm_category

def get_detected_disciplines2(input_data=None, threshold=None, declared_dsp=None):
    import re
    from gensim.similarities import SparseTermSimilarityMatrix
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel
    from gensim.similarities import WordEmbeddingSimilarityIndex
    import gensim.downloader as api
    from gensim.similarities import SoftCosineSimilarity
    import numpy as np

    text = input_data["Translated"].to_list()
    clean_text = preprocess_text(" ".join(text))
    clean_text = clean_text.replace("\n"," ")

    glove = api.load("glove-wiki-gigaword-50")    
    similarity_index = WordEmbeddingSimilarityIndex(glove)


    #key words fro disciplines
    dsp2 = {'ageing': 'ageing,programmed,animals,humans,whereby,dna,die,damage,accumulation,whereas,changes,cause,age,may,concept,disease,example,time,many,uncertain',
    'architecture': 'architecture,architectural,architects,buildings,began,building,century,forms,ancient,utility,surviving,style,put,perceived,later,idea,designing,construction,beauty,architectura',
    'behavioural sciences': 'behavioral,observation,organisms,cognitive,behavior,sciences,relates,psychobiology,naturalistic,legitimate,generalize,formulations,experimentation,controlled,attempts,animal,society,rigorous,modeling,examples',
    'classical studies': 'classics,roman,traditionally,classical,greek,western,study,original,mythology,greco,elite,typical,therefore,languages,foundation,cornerstone,civilization,archaeology,antiquity,refers',
    'communication sciences': 'communication,information,social,interpersonal,face,agency,media,studies,techniques,individual,activity,focus,range,level,knowledge,scientific,methods,public,analysis,cultural',
    'community health': 'health,community,services,care,volunteers,healthcare,local,registered,providers,officers,assistants,primary,preventive,treatment,workers,provide,members,supervisory,rehabilitative,qualifications',
    'cultural heritage': 'heritage,cultural,protection,tangible,landscapes,intangible,generations,english,property,past,nations,culture,international,united,unesco,traditions,tourism,shield,selection,relating',
    'museology': 'museums,programming,museum,museology,explores,engage,curating,preservation,activities,role,well,history,public,studies,education,society,including,study',
    'demography': 'demographic,patient,demography,populations,population,census,analysis,independent,da,birth,death,information,data,medical,migration,estimate,emergency,demographics,contact,2010',
    'development studies': 'development,organisations,countries,studies,world,uk,responsibility,researched,popularity,ngos,grown,csr,consultancy,colonial,choose,careers,bank,reputed,offered,journalism',
    'economics': 'economics,agents,economic,interactions,applied,production,consumption,analyzes,elements,economy,individual,whats,sellers,saving,outcomes,ought,normative,microeconomics,markets,mainstream',
    'finances': 'finance,financial,money,assets,economics,minimize,maximize,accounting,journal,action,discipline,value,programs,early,system,fields,century,academic,history,volatility',
    'education': 'education,formal,educational,informal,based,skills,aims,may,structured,occurs,nonformal,achieving,one,training,teaching,learning,goals,takes,settings,school',
    'environmental studies': 'environmental,degree,studies,environment,tools,ethics,issues,principles,programs,address,field,planning,natural,economics,sciences,systematically,resource,raise,pro,pollution',
    'game design': 'game,games,design,playing,mechanics,rules,principles,elements,theory,systems,research,studies,zubek,users,theyre,systemsmechanics,situations,simulation,shaping,seminal',
    'gender studies': 'gender,studies,theory,disciplines,sometimes,field,rise,however,approaches,political,politics,psychology,fields,sociology,many,history,anthropology,womens,women,view',
    'geography': 'earth,geography,things,greek,everything,af,first,related,discipline,field,waldo,tobler,recorded,planetary,near,merely,lands,inhabitants,graphien,geographia',
    'health promotion': 'health,promotion,stated,ottawa,1986,improve,enabling,charter,increase,control,organization,process,world,people',
    'health psychology': 'health,psychology,psychologists,psychological,behavioral,factors,clinical,scale,harm,four,division,affect,patients,divisions,american,processes,working,professionals,illness,public',
    'history': 'history,events,historians,past,bc,study,modern,thucydides,narrative,herodotus,father,continue,historical,supported,sources,nature,helped,debate,well,writing',
    'international relations': 'international,relations,war,ir,organisations,concerns,states,interactions,political,politics,major,union,trade,subsequent,soviet,sovereign,second,scholarship,response,rapidly',
    'law': 'law,jurisdictions,legal,legislature,common,may,private,precedent,judges,disputes,court,contracts,binding,religious,justice,civil,ways,influenced,laws,countries',
    'linguistics': 'language,linguistics,theoretical,cognitive,scientific,structure,abstract,study,traditional,practical,nature,describing,linguistic,aspects,meaning,humanities,developing,concerned,social,science',
    'literature': 'literature,works,prose,poetry,letters,fiction,writing,drama,written,definition,according,also,non,form,includes,art,term,writings,transcribed,sung',
    'management': 'management,managers,organization,business,line,administration,organizations,senior,master,direction,front,resources,provide,may,perform,managing,bachelor,volunteers,strategic,roles',
    'medical anthropology': 'anthropology,health,illness,care,medical,term,issues,used,studies,cultural,views,subfield,representations,nursing,multidimensional,medische,label,dutch,chosen,biocultural',
    'methods and statistics': 'data,null,hypothesis,statistical,statistics,errors,two,sample,sets,relationship,random,measurements,false,using,collection,population,variation,test,taking,sampling',
    'musicology and performing arts': 'musicology,music,musical,research,musicologists,interest,computational,traditionally,study,cognitive,systematic,historical,history,field,sociological,regarded,proper,physiology,participates,origin, arts,performing,dance,music,performed,audience,theatre,performances,live,stages,date,visual,theatres,tents,street,static,professionally,paint,opera,open',
    'philosophy and sociology of sciences': 'social,sciences,similarities,possible,philosophers,ontological,logic,existence,significance,foundations,differences,causal,science,etc,agency,laws,concerned,structure,relationships,phenomena',
    'political science': 'political,science,structuralism,politics,subdisciplines,sources,research,theory,psychology,philosophy,methods,social,analysis,positivism,pluralism,originating,official,notable,methodologically,interpretivism',
    'psychology': 'psychologists,involved,psychology,behavior,settings,mental,unconscious,resilience,mind,counseling,scientists,problems,others,research,discipline,group,functions,cognitive,clinical,social',
    'public health epidemiology': 'epidemiology,disease,epidemiologists,epidemic,study,upon,population,populations,term,review,plant,endemic,conditions,causation,better,diseases,used,health,widely,clinical,health,public,care,countries,initiatives,developing,disease,prevention,epidemiology,cases,promoting,diseases,sciences,promotion,healthcare,control,communities,population,example,level',
    'public health nutrition': 'nutrients,nutrition,consuming,organisms,obtain,energy,absorbing,require,organism,matter,life,support,soil,proteins,produce,organic,obtains,mycelium,must,molecules,nutrients,nutrition,consuming,organisms,obtain,energy,absorbing,require,organism,matter,life,support,soil,proteins,produce,organic,obtains,mycelium,must,molecules',
    'qualitative methodology': 'research,qualitative,want,gain,experiences,peoples,type,data,order,understanding,methods,used,analysis,useful,underlying,uncover,software,rich,reasons,reality',
    'social anthropology and ethnology': 'ethnos,ethnology,compares,compare,characteristics,sociocultural,nation,peoples,analyzes,meaning,relationships,greek,different,academic,anthropology,field,cultural,social,anthropology,united,cultural,subsumed,dominant,constituent,sociocultural,kingdom,distinguished,behaviour,throughout,much,cultures,europe,states,societies,patterns,commonly,social,within',
    'sociology': 'social,sociology,research,analyses,focuses,agency,analysis,organizations,expanded,society,techniques,life,interaction,individual,activity,structure,focus,range,non,level,space,outer,vehicles,traffic,debris,international,station,satellites,safe,return,regulatory,radio,provisions,orbiting,operations,mitigation,launch,interference,iaa,frequency'}  
    
     #run semantic comparison
    
    #run semantic comparison
    query_string = str(clean_text)
    documents = list(dsp2.values())

    # Preprocess the documents, including the query string
    corpus = [preprocess_sw(document) for document in documents]
    query = preprocess_sw(query_string)

    # Build the term dictionary, TF-idf model
    dictionary = Dictionary(corpus+[query])
    tfidf = TfidfModel(dictionary=dictionary)

    # Create the term similarity matrix.  
    similarity_index = WordEmbeddingSimilarityIndex(glove)
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

    # Compute Soft Cosine Measure between the query and the documents.
    query_tf = tfidf[dictionary.doc2bow(query)]

    index = SoftCosineSimilarity(
                tfidf[[dictionary.doc2bow(document) for document in corpus]],
                similarity_matrix)

    doc_similarity_scores = index[query_tf]

    # Output the sorted similarity scores and documents
    sorted_indexes = np.argsort(doc_similarity_scores)[::-1]

    samples = []
    for idx in sorted_indexes:
        if doc_similarity_scores[idx] > threshold:
            samples.append(documents[idx])

    #create dictionary from list of detected disciplines
    counts = dict()
    target_dsp = [x.lower().strip() for x in set(declared_dsp)]

    for idx in sorted_indexes:
        if doc_similarity_scores[idx] > 0.25:
            print((list(dsp2.keys())[idx]))
            if list(dsp2.keys())[idx] in target_dsp:
                counts.update({list(dsp2.keys())[idx] : doc_similarity_scores[idx]})


    return counts

def get_detected_disciplines_new(ibm_category=None, threshold=None,list_discipline=None):
    from gensim.similarities import SparseTermSimilarityMatrix
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel
    from gensim.similarities import WordEmbeddingSimilarityIndex
    import gensim.downloader as api
    from gensim.similarities import SoftCosineSimilarity
    import numpy as np

    glove = api.load("glove-wiki-gigaword-50")    

    list_of_disciplines = list_discipline
    #run semantic comparison
    query_string = str(ibm_category)
    documents = list_of_disciplines

    # Preprocess the documents, including the query string
    corpus = [preprocess(document) for document in documents]
    query = preprocess(query_string)

    # Build the term dictionary, TF-idf model
    dictionary = Dictionary(corpus+[query])
    tfidf = TfidfModel(dictionary=dictionary)

    # Create the term similarity matrix.  
    similarity_index = WordEmbeddingSimilarityIndex(glove)
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

    # Compute Soft Cosine Measure between the query and the documents.
    query_tf = tfidf[dictionary.doc2bow(query)]

    index = SoftCosineSimilarity(
                tfidf[[dictionary.doc2bow(document) for document in corpus]],
                similarity_matrix)

    doc_similarity_scores = index[query_tf]

    # Output the sorted similarity scores and documents
    sorted_indexes = np.argsort(doc_similarity_scores)[::-1]

    samples = []
    for idx in sorted_indexes:
        if doc_similarity_scores[idx] > threshold:
            samples.append(documents[idx])

    #create dictionary from list of detected disciplines
    counts = dict()
    for i in samples:
        counts[i] = counts.get(i, 0) + 1

    return counts

def get_n_counts(counts, k=28):
    return balance(counts,k)


############################################
# discipline detected in messages
############################################
def get_ibm_category_messages(input_data =None, api_key=None, filename = None):

    # Load the model: this is a big file, can take a while to download and open
    import json
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, ConceptsOptions, CategoriesOptions, EntitiesOptions

    import pandas as pd 
    data = pd.read_csv(input_data)
    text = data["Body"].to_list()
    text = [x for x in text if pd.isna(x) ==False]
    text = " ".join(text)
    authenticator = IAMAuthenticator(api_key)  #insert here your NLU API key
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )

    natural_language_understanding.set_service_url('https://api.eu-de.natural-language-understanding.watson.cloud.ibm.com/instances/8d485964-c1f1-4001-b9ba-09f99451baca')

    response = natural_language_understanding.analyze(
    text = text,  #insert here URL to Blogs (better option will be to generate one mixted URL for all articles)
    features=Features(categories=CategoriesOptions(limit=1))).get_result()

    with open(filename + "_categories_messages.json", "w") as out_file:
        json.dump(response, out_file, indent=2)

    with open(filename + "_categories_messages.json") as json_file:
        data = json.load(json_file)
        for p in data['categories']:
            category = p['label']
            ibm_category = category.replace("/", " ")
    return ibm_category

def get_ibm_category_messages_from_json(filename = None):

    # Load the model: this is a big file, can take a while to download and open
    import json

    import pandas as pd 
    with open(filename + "_categories_messages.json") as json_file:
        data = json.load(json_file)
        for p in data['categories']:
            category = p['label']
            ibm_category = category.replace("/", " ")
    return ibm_category


#################################
# PLOT #
#############

def plt_discipline(n_blog_counts, n_message_counts, outdir):
    #two the subindicators on one figure
    # The asymmetry for these 2 subindicators is measured in reversed way: 0 - high asymmetry, 1 - low. It means when we detect more disciplines we reach the symmetry.
    import plotly.graph_objects as go
    # Add data
    month = list(n_message_counts.keys())
    discp_blog = list(n_blog_counts.values())
    discp_emails = list(n_message_counts.values())


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=month, y=discp_emails, name='Changes in disciplines detected in e-mail exchanges ',
                            line=dict(color='orange', width=4)))
    fig.add_trace(go.Scatter(x=month, y=discp_blog, name = 'Changes in disciplines detected in deliverables',
                            line=dict(color='blue', width=4)))
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
    # Edit the layout
    fig.update_layout(
        #title='<b>Change of Asymmetry Sub-indicators <br>Scale (min 0, max 1)</b>',
                    xaxis_title='Month',
                    yaxis_title='Score')

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i><br> Scale: <br> 0 = high asymmetry <br> 1 = low asymmetry<i>")
        , showarrow=False
        , x = 0
        , y = -0.25
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )

    fig.write_image(outdir  +"\\discipline_blogs_messages.png")

#############
# discipline in blog
####################
def get_detected_disciplines_blog(input_data=None, threshold=None, declared_dsp=None):
    import re
    from gensim.similarities import SparseTermSimilarityMatrix
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel
    from gensim.similarities import WordEmbeddingSimilarityIndex
    import gensim.downloader as api
    from gensim.similarities import SoftCosineSimilarity
    import numpy as np
    text = open(input_data, "r").read()
    import pandas as pd

    #Translate text to English
    # t_text=translator.translate(text,dest='en').text
    
    clean_text = preprocess_text(text)
    clean_text = clean_text.replace("\n"," ")

    glove = api.load("glove-wiki-gigaword-50")    
    similarity_index = WordEmbeddingSimilarityIndex(glove)


    #key words fro disciplines
    dsp2 = {'ageing': 'ageing,programmed,animals,humans,whereby,dna,die,damage,accumulation,whereas,changes,cause,age,may,concept,disease,example,time,many,uncertain',
    'architecture': 'architecture,architectural,architects,buildings,began,building,century,forms,ancient,utility,surviving,style,put,perceived,later,idea,designing,construction,beauty,architectura',
    'behavioural sciences': 'behavioral,observation,organisms,cognitive,behavior,sciences,relates,psychobiology,naturalistic,legitimate,generalize,formulations,experimentation,controlled,attempts,animal,society,rigorous,modeling,examples',
    'classical studies': 'classics,roman,traditionally,classical,greek,western,study,original,mythology,greco,elite,typical,therefore,languages,foundation,cornerstone,civilization,archaeology,antiquity,refers',
    'communication sciences': 'communication,information,social,interpersonal,face,agency,media,studies,techniques,individual,activity,focus,range,level,knowledge,scientific,methods,public,analysis,cultural',
    'community health': 'health,community,services,care,volunteers,healthcare,local,registered,providers,officers,assistants,primary,preventive,treatment,workers,provide,members,supervisory,rehabilitative,qualifications',
    'cultural heritage': 'heritage,cultural,protection,tangible,landscapes,intangible,generations,english,property,past,nations,culture,international,united,unesco,traditions,tourism,shield,selection,relating',
    'museology': 'museums,programming,museum,museology,explores,engage,curating,preservation,activities,role,well,history,public,studies,education,society,including,study',
    'demography': 'demographic,patient,demography,populations,population,census,analysis,independent,da,birth,death,information,data,medical,migration,estimate,emergency,demographics,contact,2010',
    'development studies': 'development,organisations,countries,studies,world,uk,responsibility,researched,popularity,ngos,grown,csr,consultancy,colonial,choose,careers,bank,reputed,offered,journalism',
    'economics': 'economics,agents,economic,interactions,applied,production,consumption,analyzes,elements,economy,individual,whats,sellers,saving,outcomes,ought,normative,microeconomics,markets,mainstream',
    'finances': 'finance,financial,money,assets,economics,minimize,maximize,accounting,journal,action,discipline,value,programs,early,system,fields,century,academic,history,volatility',
    'education': 'education,formal,educational,informal,based,skills,aims,may,structured,occurs,nonformal,achieving,one,training,teaching,learning,goals,takes,settings,school',
    'environmental studies': 'environmental,degree,studies,environment,tools,ethics,issues,principles,programs,address,field,planning,natural,economics,sciences,systematically,resource,raise,pro,pollution',
    'game design': 'game,games,design,playing,mechanics,rules,principles,elements,theory,systems,research,studies,zubek,users,theyre,systemsmechanics,situations,simulation,shaping,seminal',
    'gender studies': 'gender,studies,theory,disciplines,sometimes,field,rise,however,approaches,political,politics,psychology,fields,sociology,many,history,anthropology,womens,women,view',
    'geography': 'earth,geography,things,greek,everything,af,first,related,discipline,field,waldo,tobler,recorded,planetary,near,merely,lands,inhabitants,graphien,geographia',
    'health promotion': 'health,promotion,stated,ottawa,1986,improve,enabling,charter,increase,control,organization,process,world,people',
    'health psychology': 'health,psychology,psychologists,psychological,behavioral,factors,clinical,scale,harm,four,division,affect,patients,divisions,american,processes,working,professionals,illness,public',
    'history': 'history,events,historians,past,bc,study,modern,thucydides,narrative,herodotus,father,continue,historical,supported,sources,nature,helped,debate,well,writing',
    'international relations': 'international,relations,war,ir,organisations,concerns,states,interactions,political,politics,major,union,trade,subsequent,soviet,sovereign,second,scholarship,response,rapidly',
    'law': 'law,jurisdictions,legal,legislature,common,may,private,precedent,judges,disputes,court,contracts,binding,religious,justice,civil,ways,influenced,laws,countries',
    'linguistics': 'language,linguistics,theoretical,cognitive,scientific,structure,abstract,study,traditional,practical,nature,describing,linguistic,aspects,meaning,humanities,developing,concerned,social,science',
    'literature': 'literature,works,prose,poetry,letters,fiction,writing,drama,written,definition,according,also,non,form,includes,art,term,writings,transcribed,sung',
    'management': 'management,managers,organization,business,line,administration,organizations,senior,master,direction,front,resources,provide,may,perform,managing,bachelor,volunteers,strategic,roles',
    'medical anthropology': 'anthropology,health,illness,care,medical,term,issues,used,studies,cultural,views,subfield,representations,nursing,multidimensional,medische,label,dutch,chosen,biocultural',
    'methods and statistics': 'data,null,hypothesis,statistical,statistics,errors,two,sample,sets,relationship,random,measurements,false,using,collection,population,variation,test,taking,sampling',
    'musicology and performing arts': 'musicology,music,musical,research,musicologists,interest,computational,traditionally,study,cognitive,systematic,historical,history,field,sociological,regarded,proper,physiology,participates,origin, arts,performing,dance,music,performed,audience,theatre,performances,live,stages,date,visual,theatres,tents,street,static,professionally,paint,opera,open',
    'philosophy and sociology of sciences': 'social,sciences,similarities,possible,philosophers,ontological,logic,existence,significance,foundations,differences,causal,science,etc,agency,laws,concerned,structure,relationships,phenomena',
    'political science': 'political,science,structuralism,politics,subdisciplines,sources,research,theory,psychology,philosophy,methods,social,analysis,positivism,pluralism,originating,official,notable,methodologically,interpretivism',
    'psychology': 'psychologists,involved,psychology,behavior,settings,mental,unconscious,resilience,mind,counseling,scientists,problems,others,research,discipline,group,functions,cognitive,clinical,social',
    'public health epidemiology': 'epidemiology,disease,epidemiologists,epidemic,study,upon,population,populations,term,review,plant,endemic,conditions,causation,better,diseases,used,health,widely,clinical,health,public,care,countries,initiatives,developing,disease,prevention,epidemiology,cases,promoting,diseases,sciences,promotion,healthcare,control,communities,population,example,level',
    'public health nutrition': 'nutrients,nutrition,consuming,organisms,obtain,energy,absorbing,require,organism,matter,life,support,soil,proteins,produce,organic,obtains,mycelium,must,molecules,nutrients,nutrition,consuming,organisms,obtain,energy,absorbing,require,organism,matter,life,support,soil,proteins,produce,organic,obtains,mycelium,must,molecules',
    'qualitative methodology': 'research,qualitative,want,gain,experiences,peoples,type,data,order,understanding,methods,used,analysis,useful,underlying,uncover,software,rich,reasons,reality',
    'social anthropology and ethnology': 'ethnos,ethnology,compares,compare,characteristics,sociocultural,nation,peoples,analyzes,meaning,relationships,greek,different,academic,anthropology,field,cultural,social,anthropology,united,cultural,subsumed,dominant,constituent,sociocultural,kingdom,distinguished,behaviour,throughout,much,cultures,europe,states,societies,patterns,commonly,social,within',
    'sociology': 'social,sociology,research,analyses,focuses,agency,analysis,organizations,expanded,society,techniques,life,interaction,individual,activity,structure,focus,range,non,level,space,outer,vehicles,traffic,debris,international,station,satellites,safe,return,regulatory,radio,provisions,orbiting,operations,mitigation,launch,interference,iaa,frequency'}  
    
     #run semantic comparison
    
    #run semantic comparison
    query_string = str(clean_text)
    documents = list(dsp2.values())

    # Preprocess the documents, including the query string
    corpus = [preprocess_sw(document) for document in documents]
    query = preprocess_sw(query_string)

    # Build the term dictionary, TF-idf model
    dictionary = Dictionary(corpus+[query])
    tfidf = TfidfModel(dictionary=dictionary)

    # Create the term similarity matrix.  
    similarity_index = WordEmbeddingSimilarityIndex(glove)
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

    # Compute Soft Cosine Measure between the query and the documents.
    query_tf = tfidf[dictionary.doc2bow(query)]

    index = SoftCosineSimilarity(
                tfidf[[dictionary.doc2bow(document) for document in corpus]],
                similarity_matrix)

    doc_similarity_scores = index[query_tf]

    # Output the sorted similarity scores and documents
    sorted_indexes = np.argsort(doc_similarity_scores)[::-1]

    samples = []
    for idx in sorted_indexes:
        if doc_similarity_scores[idx] > threshold:
            samples.append(documents[idx])

    #create dictionary from list of detected disciplines
    counts = dict()
    target_dsp = [x.lower().strip() for x in set(declared_dsp)]
    for idx in sorted_indexes:
        if doc_similarity_scores[idx] > threshold:
            if list(dsp2.keys())[idx] in target_dsp:
                counts.update({list(dsp2.keys())[idx] : doc_similarity_scores[idx]})


    return counts