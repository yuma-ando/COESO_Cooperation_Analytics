#!/usr/bin/env python
# coding: utf-8

##################################
# data extraction and cleaning
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
        
###########################################
### Subindicator 
############################################


# all the imports
import statistics
from statistics import median
import pandas as pd 
from datetime import datetime
from math import nan, isnan
import re
import math
import io
import plotly.graph_objects as go
import calendar

# # Getting the data

# Please use `[Tool_for_extracting_filtering_and_cleaning_GMAIL_data.ipynb](https://colab.research.google.com/drive/18jBP1XBQ06AkUyb4pEKwcMf4_ePADOeA?usp=share_link)` for extracting data from mbox file and saving it to csv format.

# In[ ]:


# data pulled from google takeot that was undergone the prepocessing step from above. We are working with csv files using pandas library.
# for running this notebook, please save Pilot7 - wp5.csv' file locally from this (Intensity) folder and choose it when interaction window will ask you to choose the file.


# upload csv file from PC
#from google.colab import files
 ##uploaded = files.upload()



#############################################################
##Common functions ###
#################################################################
def extract_sender(data=None):
    all_from = data["From"].values
    # remove nan values
    mail_from_without_nan = [x for x in all_from if str(x) != 'nan']

    # make a string from list of people names and their mails
    all_values_from = " ".join(mail_from_without_nan)

    # extract email addresses from names and emails by pattern for email adresses identification
    match_values_from = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', all_values_from)

    # all the unique sender emails from data dataframe we are goint to iterate over 

    sender_emails = [item for item in set(match_values_from)]
    sender_emails

    return sender_emails




# Extract all the senders emails: sender email == project participant



# all the unique sender emails from data dataframe we are goint to iterate over 

# # Helper functions



#Transform total number of seconds into "days hh:mm:ss" format

def seconds_to_hr_min_sec(seconds):
    import datetime
    if seconds:
        return str(datetime.timedelta(seconds=seconds))
    else:
        return None
 # Example:
seconds_to_hr_min_sec(666)


# convert body of message in one str without spaces and
# remove ">" symbol for further searching replies in messages without Subject

def convert_to_raw_str(body):

    l = [word for word in body.split()]
    raw_body = "".join(l).replace(">", "")
    return raw_body


# In[11]:


# function to modify dict keys dates converted to string into datetime

import dateutil

def modify_datetime(daytime):
    tzmapping = {'CET': dateutil.tz.gettz('Europe/Berlin')}
    dtobj = dateutil.parser.parse(daytime, tzinfos=tzmapping)

    return dtobj


# In[12]:


# convert time if format  '1 day, 01:00:00' to hours

def get_hours(time_1):
    import datetime
    import re

#     time_1 = '1 day, 01:00:00'

    for (days, t) in re.findall(r'(?:(\d+)\s+days?)?,?\s*([\d:]+)', time_1):
        h, m, s = str(t).split(':')
        seconds = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s)).total_seconds()
        hours = seconds/3600
        if days:
            seconds = seconds + datetime.timedelta(days=int(days)).total_seconds()
            hours = seconds/3600
#         print(int(seconds))
        return round(hours, 2)


# In[13]:


# convert time in hh:mm:ss:' format to total seconds

def get_seconds(time_str):
    print('Time in hh:mm:ss:', time_str)
    # split in hh, mm, ss
    hh, mm, ss = time_str.split(':')
    return int(hh) * 60 + int(mm) * 60 + int(ss)



# function to remove prefixes in subjects

def remove_prefixes_in_sbj(subject):
    import re
    p = re.compile( '([\[\(] *)?(RE?S?|FYI|RIF|I|FS|VB|RV|ENC|ODP|PD|YNT|ILT|SV|VS|VL|AW|WG|ΑΠ|ΣΧΕΤ|ΠΡΘ|תגובה|הועבר|主题|转发|FWD?) *([-:;)\]][ :;\])-]*|$)|\]+ *$', re.IGNORECASE)
    return p.sub( '', subject).strip()





#############################################################
##Delays in responses ###
#################################################################


# Core function for computiong delays in responses
# Actually meadian instead of mean response delay was used here to decrease the impact of outliers

def find_mean_answer_delay_per_user_new(email, df, data = None):
    
    email_to_subj_date = dict()
    for index, row in df.iterrows():
        if email in str(row["From"]):
            if email not in email_to_subj_date:
                
                email_to_subj_date[email]=[{row["Subject"]:row["Date_Converted_dt"]}\
                                           if str(row["Subject"])!= "nan"\
                                           else\
                                           {row["Subject"]:[row["Date_Converted_dt"],\
                                                            convert_to_raw_str(str(row["Body"]))]}]
            else:
                email_to_subj_date[email].append([{row["Subject"]:row["Date_Converted_dt"]} \
                                                  if str(row["Subject"])!= "nan" \
                                                  else\
                                                  {row["Subject"]:[row["Date_Converted_dt"],\
                                                                   convert_to_raw_str(str(row["Body"]))]}][0])

    reply_mail_subjects_to_date = [dic for dic in email_to_subj_date[email]]

    time_differ_in_all_repl_by_user = []
    only_no_subj_delays = []
    re_date_to_match_income_mail_dates=dict()
    for subj_date in reply_mail_subjects_to_date:
        for index, row in data.iterrows():
            # check messages with no subject:
            if str(list(subj_date.keys())[0])=="nan":
                if str(row["Subject"]) == "nan" and str(row["Body"])!='nan'\
                    and email in str(row["To"]) and\
                    list(subj_date.values())[0][0] != row["Date_Converted_dt"] and\
                    convert_to_raw_str(str(row["Body"])) in list(subj_date.values())[0][1]:
                    
                # add time delay calculated for messages with no Subject
                  
                    no_sbj_delay = (list(subj_date.values())[0][0] -\
                                                          row["Date_Converted_dt"]).total_seconds()
                    if no_sbj_delay:
                        time_differ_in_all_repl_by_user.append(no_sbj_delay)
                        only_no_subj_delays.append(no_sbj_delay)

            else:
                if str(list(subj_date.keys())[0]) == str(row["Subject"]) and\
                    email in str(row["To"])\
                    and list(subj_date.values())[0] != row["Date_Converted_dt"]:

                    if str(list(subj_date.values())[0]) not in re_date_to_match_income_mail_dates:
                        re_date_to_match_income_mail_dates[str(list(subj_date.values())[0])]=[row["Date_Converted_dt"]]
                    else:
                        re_date_to_match_income_mail_dates[str(list(subj_date.values())[0])].append(row["Date_Converted_dt"])


    for tup in re_date_to_match_income_mail_dates.items():
        time_differ_in_repl = []
        for value in tup[1]:
            if value < modify_datetime(tup[0]):
                time_differ_in_repl.append((modify_datetime(tup[0]) - value).total_seconds())
        if time_differ_in_repl:
            time_differ_in_all_repl_by_user.append(min(time_differ_in_repl))
        
    if time_differ_in_all_repl_by_user:
        average_response_delay_for_user = median(time_differ_in_all_repl_by_user)
    else:
        average_response_delay_for_user = []


    return average_response_delay_for_user


# In[17]:


# iterate over all the mails to find mean response delay in sec (clean version)
def get_users_answer_delays_new(sender_emails, df, data):
    seconds_per_user = []
    all_users_delays = {}
    for email in sender_emails:
        delay = find_mean_answer_delay_per_user_new(email, df, data)
        if delay:
            seconds_per_user.append(delay) 
            all_users_delays[email] = seconds_to_hr_min_sec(delay)
        else:
            all_users_delays[email] = []
    if seconds_per_user:
        mean_delay_per_project_seconds = round(sum(seconds_per_user)/len(seconds_per_user))
    else:
        mean_delay_per_project_seconds = None
    print("Mean delay per project in seconds:", mean_delay_per_project_seconds)
    mean_delay_per_project_day_time = seconds_to_hr_min_sec(mean_delay_per_project_seconds)
    print("Mean delay per project, day/time:", seconds_to_hr_min_sec(mean_delay_per_project_seconds))

    return mean_delay_per_project_day_time, all_users_delays





def read_data(input_data=None):
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
    data = data.sort_values('Date_Converted_dt', ascending=True) ###sort all data by Date_Converted_dt values

    # apply removing replies prefixes in the subjects:
    data["Subject"] = data["Subject"].astype('str').apply(remove_prefixes_in_sbj)
    
    # convert Date_Converted_dt to datetime in order to sort df by months
    data["Date_Converted_dt"] = pd.to_datetime(data.Date_Converted_dt, utc=True)

    return data




# try running function on a separate email --> returns dict(email: average delay in seconds)  
# Consider all the provided emails belong to 1 month
#mean_delay_for_project, result = get_users_answer_delays_new(sender_emails, data)


#def data_by_month(data=None):
#    available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
#    dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]
#    return dfs

# check for which months the data was provided: list of order number of month
    

# result: 8 = August, 9  =September



def get_delays(data = None):
    #available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
    #print("available months are " , available_months)
    #dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]

    available_months = sorted(list(data.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [data[data.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]

    delays = dict()
    for i, df in zip(available_months, dfs):
        print(f"Month # {i}:\n")
        print("extraction sendors")
        sender_emails = extract_sender(df)
        print("senders extracted")
        try:
            mean_delay_for_project, result = get_users_answer_delays_new(sender_emails, df, data)
            #print(result)
            #delays.update({calendar.month_name[i] : mean_delay_for_project})
            delays.update({str(i) : mean_delay_for_project})

        except KeyError:
            print(f"key error for month {i}")
            continue
    return delays


#############################################################
##delays in response : final function ###
#################################################################

def delay_in_email_responses(input = None):

    data = read_data(input_data=input)
#    dfs = data_by_month(data=data)
#    sender_emails = extract_sender(data)
#    delays = get_delays(data =data, sender_emails = sender_emails)
    delays = get_delays(data =data)

    return delays


#############################################################
##Mean number of received messages ###
#################################################################
def get_mean_number_of_received_messages(data):
    from math import nan, isnan
    import re

    # all the values From
    all_to = data["To"].values
    # remove nan values
    mail_to_without_nan = [x for x in all_to if str(x) != 'nan']
#     print(len(mail_to_without_nan))

    # make a string from list of people names and their mails
    all_values_to = " ".join(mail_to_without_nan)

    # extract email addresses from names and emails by pattern for email adresses identification
    match_values_to = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', all_values_to)

    # unique email addresses to count of sent messages by that email address
    count_values_to = {el:match_values_to.count(el) for el in set(match_values_to)}

    sorted_count_values_to = sorted(count_values_to.items(), key=lambda x: x[1], reverse=True) # descending order
    mean_value_of_received_messages_per_user = round(sum([el[1] for el in sorted_count_values_to])/len(sorted_count_values_to))
#     print(mean_value_of_received_messages_per_user)
    return mean_value_of_received_messages_per_user

def get_mean_received(input = None):
    data = read_data(input_data=input)
    #available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
    #print("available months are " , available_months)
    #dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]

    available_months = sorted(list(data.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [data[data.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]

    mean_received = dict()
    for i, df in zip(available_months, dfs):
        print(f"Month # {i}:\n")
        try:
            #print(result)
            #mean_received.update({calendar.month_name[i] : get_mean_number_of_received_messages(df)})
            mean_received.update({str(i) : get_mean_number_of_received_messages(df)})

        except KeyError:
            print(f"key error for month {i}")
            continue
    return mean_received


#############################################################
##Mean number of received messages ###
#################################################################

def get_mean_number_of_sent_messages(data):

    # all the values From
    all_from = data["From"].values
    # remove nan values
    mail_from_without_nan = [x for x in all_from if str(x) != 'nan']
#     print(len(mail_from_without_nan))

    # make a string from list of people names and their mails
    all_values_from = " ".join(mail_from_without_nan)

    # extract email addresses from names and emails by pattern for email adresses identification
    match_values_from = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', all_values_from)

    # unique email addresses to count of sent messages by that email address
    count_values_from = {el:match_values_from.count(el) for el in set(match_values_from)}

    sorted_count_values_from = sorted(count_values_from.items(), key=lambda x: x[1], reverse=True) # descending order
    mean_value_of_sent_messages_per_user = round(sum([el[1] for el in sorted_count_values_from])/len(sorted_count_values_from))
#     print(mean_value_of_sent_messages_per_user)
    return mean_value_of_sent_messages_per_user

def get_mean_sent(input = None):
    data = read_data(input_data=input)
    #available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
    #print("available months are " , available_months)
    #dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]
    available_months = sorted(list(data.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [data[data.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]

    mean_sent = dict()
    for i, df in zip(available_months, dfs):
        print(f"Month # {i}:\n")
        try:
            #print(result)
            #mean_sent.update({calendar.month_name[i] : get_mean_number_of_sent_messages(df)})
            mean_sent.update({str(i): get_mean_number_of_sent_messages(df)})

        except KeyError:
            print(f"key error for month {i}")
            continue
    return mean_sent

#############################################################
##Total number of send + received###
#################################################################



def get_total_sent(data):

    # all the values From
    all_from = data["From"].values
    # remove nan values
    mail_from_without_nan = [x for x in all_from if str(x) != 'nan']

    # make a string from list of people names and their mails
    all_values_from = " ".join(mail_from_without_nan)

    # extract email addresses from names and emails by pattern for email adresses identification
    match_values_from = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', all_values_from)

    # unique email addresses to count of sent messages by that email address
    count_values_from = {el:match_values_from.count(el) for el in set(match_values_from)}

    sorted_count_values_from = sorted(count_values_from.items(), key=lambda x: x[1], reverse=True) # descending order
#     print("sorted counts of values From:",sorted_count_values_from)

    total_sent = sum([el[1] for el in sorted_count_values_from])
    return total_sent

def get_total_received(data):
    from math import nan, isnan
    import re

    # all the values From
    all_to = data["To"].values
    # remove nan values
    mail_to_without_nan = [x for x in all_to if str(x) != 'nan']
#     print(ln(mail_to_without_nan))

    # make a string from list of people names and their mails
    all_values_to = " ".join(mail_to_without_nan)

    # extract email addresses from names and emails by pattern for email adresses identification
    match_values_to = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', all_values_to)

    # unique email addresses to count of sent messages by that email address
    count_values_to = {el:match_values_to.count(el) for el in set(match_values_to)}
    sorted_count_values_to = sorted(count_values_to.items(), key=lambda x: x[1], reverse=True) # descending order
#     print("sorted counts of values To:",sorted_count_values_to)

    total_received = sum([el[1] for el in sorted_count_values_to])
    return total_received


def get_total_messag(input= None):
    data = read_data(input_data=input)
    #available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
	#dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]

    available_months = sorted(list(data.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [data[data.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]

    total_messag = dict()
    for i, df in zip(available_months, dfs):
        print(f"Month # {i}:\n")
        try:
            #print(result)
            #total_messag.update({calendar.month_name[i] : get_total_sent(df)+get_total_received(df)})
            total_messag.update({str(i): get_total_sent(df)+get_total_received(df)})

        except KeyError:
            print(f"key error for month {i}")
            continue
    return total_messag

#########################################
### Normalization 
##################################

def normalize_with_rolling_max(months, values):
    max_v = max(values)
    normalized = [(round(value/max_v, 2)) for value in values]
    print(values)
    print(normalized)
    d = {k:v for k, v in zip(months,normalized)}
    return d

def normalize_from_stored_data(stored_data, selected_column, target_column):
    import re
    col_months = [re.findall(r'\d{4}-\d{2}',x)[0] for x in stored_data.columns.tolist() if bool(re.search(r'\d{4}-\d{2}',x))]
    monthly_data= stored_data.loc[stored_data.Indicator==selected_column, col_months]
    monthly_data = monthly_data.dropna(axis=1)
    monthly_dic = {col : value.values[0] for col, value in monthly_data.iteritems()}
    months = list(monthly_dic.keys())
    values = list(monthly_dic.values())
    normalized = normalize_with_rolling_max(months, values)
    
    stored_data.loc[stored_data.Indicator==target_column, monthly_data.columns] =list(normalized.values())

def normalize_from_stored_data_delay(stored_data, selected_column, target_column):
    import re
    col_months = [re.findall(r'\d{4}-\d{2}',x)[0] for x in stored_data.columns.tolist() if bool(re.search(r'\d{4}-\d{2}',x))]
    monthly_data= stored_data.loc[stored_data.Indicator==selected_column, col_months]
    monthly_data = monthly_data.dropna(axis=1)
    if monthly_data.empty : 
        print("No delay data")
        return
    monthly_dic = {col : get_hours(value.values[0]) for col, value in monthly_data.iteritems()}
    months = list(monthly_dic.keys())
    values = list(monthly_dic.values())
    normalized = normalize_with_rolling_max(months, values)
    normalized= {col: 1-v for col, v in normalized.items() }
    stored_data.loc[stored_data.Indicator==target_column, monthly_data.columns] =list(normalized.values())



def get_hours(time_1):
    import datetime
    import re

    if time_1 is None: 
        rv = None

    for (days, t) in re.findall(r'(?:(\d+)\s+days?)?,?\s*([\d:]+)', time_1):
        h, m, s = str(t).split(':')
        seconds = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s)).total_seconds()
        hours = seconds/3600
        if days:
            seconds = seconds + datetime.timedelta(days=int(days)).total_seconds()
            hours = seconds/3600
#         print(int(seconds))
        rv = round(hours, 2)
    return rv
        
def get_normalized_delay(delays = None):
    filtered = dict()
    for k, v in delays.items():
        if v == None : 
            #print(f"the delay for {k} is None ; removed from the calculation")
            filtered.update({k : "00:00:00" })
        if v != None : 
            filtered.update({k : v })
    

    delays_hrs = {k: get_hours(v) for k, v in filtered.items()}
    months = list(delays_hrs.keys())
    values = list(delays_hrs.values())
    delay = normalize_with_rolling_max(months, values)
    return delay

def get_normalized_mean_received(mean_received = None):
    filtered = dict()
    for k, v in mean_received.items():
        if v == None : 
            print(f"the mean_received for {k} is None ; removed from the calculation")
        if v != None : 
            filtered.update({k : v })
    

    months = list(filtered.keys())
    values = list(filtered.values())
    mean_received = normalize_with_rolling_max(months, values)
    return mean_received

    
def get_normalized_mean_sent(mean_sent = None):
    filtered = dict()
    for k, v in mean_sent.items():
        if v == None : 
            print(f"the mean_sent for {k} is None ; removed from the calculation")
        if v != None : 
            filtered.update({k : v })
    

    months = list(filtered.keys())
    values = list(filtered.values())
    mean_sent = normalize_with_rolling_max(months, values)
    return mean_sent

def get_normalized_total_messag(total_messag = None):
    filtered = dict()
    for k, v in total_messag.items():
        if v == None : 
            print(f"the total_messag for {k} is None ; removed from the calculation")
        if v != None : 
            filtered.update({k : v })
    

    months = list(filtered.keys())
    values = list(filtered.values())
    total_messag = normalize_with_rolling_max(months, values)
    return total_messag


    ####################
    ###plot
    ####################

def plt_total_messag(n_total_messag= None, outdir = None):
    import plotly.graph_objects as go
    # Add data
    month = list(n_total_messag.keys())
    scores = list(n_total_messag.values())
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Messaging_intesity_score',
        line=dict(color='firebrick', width=4,
        dash='dash'))) # dash options include 'dash', 'dot', and 'dashdot'
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )    
    # Edit the layout
    fig.update_layout(
        title='<b>Change of Total number of sent and received messages</b><br><br><i>** Each month value was normalized using rolling max principle (divided by the max value based on the previous<br>months including the current month value)<i><br>',
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
    #     , xshift=-1
    #     , yshift=-5
        , font=dict(size=13, color="grey")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    # fig.update_layout(title={'font': {'size': 18}})

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )

    fig.write_image(outdir + "\\total_message.png")
    #fig.show()

def plt_mean_received(n_mean_received=None, outdir = None) :
    import plotly.graph_objects as go
    # Add data
    month = list(n_mean_received.keys())
    scores = list(n_mean_received.values())
    fig = go.Figure()
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Messaging_intesity_score',
                            line=dict(color='purple', width=4,
                                dash='dashdot'))) # dash options include 'dash', 'dot', and 'dashdot'
    # Edit the layout
    fig.update_layout(
        title='<b>Change of Mean number of received messages</b><br><br><i>** Each month value was normalized using rolling max principle (divided by the max value based on the previous<br>months including the current month value)<i><br>',
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
    #     , xshift=-1
    #     , yshift=-5
        , font=dict(size=13, color="grey")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    # fig.update_layout(title={'font': {'size': 18}})

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.write_image(outdir  +"\\mean_received.png")
    #fig.show()

def plt_mean_sent(n_mean_sent=None, outdir = None) :
    import plotly.graph_objects as go
    # Add data
    month = list(n_mean_sent.keys())
    scores = list(n_mean_sent.values())
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Messaging_intesity_score',
                            line=dict(color='green', width=4,
                                dash='dashdot'))) # dash options include 'dash', 'dot', and 'dashdot'
    # Edit the layout
    fig.update_layout(
        title='<b>Change of Mean number of sent messages</b><br><br><i>** Each month value was normalized using rolling max principle (divided by the max value based on the previous<br>months including the current month value)<i><br>',
                    xaxis_title='Month',
                    yaxis_title='Score')
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )

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
    #     , xshift=-1
    #     , yshift=-5
        , font=dict(size=13, color="grey")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    # fig.update_layout(title={'font': {'size': 18}})

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.write_image(outdir  +"\\mean_sent.png")
    #fig.show()

def plt_delays(n_delays = None, outdir=None):
    import plotly.graph_objects as go
    # Add data
    month = list(n_delays.keys())
                                                
    # IMPORTANT!!!!

    # We should transfrom the scale from [0:1] to the common [1:0] for this negatively correlated indicator.
    # Before rescaling:
    # scores = [0.16, 1.0, 0.8, 0.11, 0.84, 0.24, 0.86, 0.48, 0.29, 0.76, 0.25]
    #scores = [round((1 - el),2) for el in l]    
    # transformed scale for Delay in email responses by subtraction of the normalized value by month from 1
    scores = list(n_delays.values())
    scores = [round((1 - el),2) for el in scores]   


    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Messaging_intesity_score',
                            line=dict(color='orange', width=4, dash="dash"))) # dash options include 'dash', 'dot', and 'dashdot'

    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )

    # Edit the layout
    fig.update_layout(
        title='<b>Change of Delay in responses</b><br><br><i>** Each month value was normalized using rolling max principle (divided by the max value based on the previous<br>months including the current month value)<i><br>',
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
    #     , xshift=-1
    #     , yshift=-5
        , font=dict(size=13, color="grey")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 18}})

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.write_image(outdir  +"\\delays_responses.png")

    #fig.show()

################################
## Messaging Intensity Macroindicator 
######################################
def get_messaging_intensity_macro(n_total_messag, n_mean_received, n_mean_sent, n_delays):
    # monthes for which we have subindicators normalized values
    months = list(n_total_messag.keys())

    # list of subindicaators dicts
    formula_sub_indicators = [n_total_messag, n_mean_received, n_mean_sent, n_delays]
    # create mapping of month to list of each subindicator value for that month
    aggregated_indicator_formula = {}
    for m in months:
        for d in formula_sub_indicators:
            if m not in aggregated_indicator_formula:
                if m in list(d.keys()):
                    aggregated_indicator_formula[m]=[d[m]]
            else:
                if m in list(d.keys()):
                    aggregated_indicator_formula[m].append(d[m])
    print(aggregated_indicator_formula)
    # Apply Messages_intensity macroindicator formula (see the description block at the beginning of the notebook)
    formula_output = [round((sum(l[:-1])-l[-1])/3, 2) for l in aggregated_indicator_formula.values()]
    print("")
    print(formula_output)
    # if computed macroindicator value for a month is lower than zero, set it as zero:
    formula_output_without_negative_values = [el if el > 0 else 0 for el in formula_output]
    print("")
    # list of final macroindicator results per month (11 values for 11 months):
    print(formula_output_without_negative_values)
    final_output = {k:v for k,v in zip(months, formula_output_without_negative_values)}
    print("")
    print(final_output)
    return final_output

def get_messaging_intensity_macro_from_stored_data(stored_data):
    # monthes for which we have subindicators normalized values
    col_months = [re.findall(r'\d{4}-\d{2}',x)[0] for x in stored_data.columns.tolist() if bool(re.search(r'\d{4}-\d{2}',x))]
    n_total_messag_df= stored_data.loc[stored_data.Indicator=="normalized_total_number_sent_received_messages", col_months]
    n_total_messag_df = n_total_messag_df.dropna(axis=1)
    n_total_messag = {col : value.values[0] for col, value in n_total_messag_df.iteritems()}

    n_mean_received_df= stored_data.loc[stored_data.Indicator=="normalized_mean_number_received_messages", col_months]
    n_mean_received_df = n_mean_received_df.dropna(axis=1)
    n_mean_received = {col : value.values[0] for col, value in n_mean_received_df.iteritems()}

    n_mean_sent_df= stored_data.loc[stored_data.Indicator=="normalized_mean_number_sent_messages", col_months]
    n_mean_sent_df = n_mean_sent_df.dropna(axis=1)
    n_mean_sent = {col : value.values[0] for col, value in n_mean_sent_df.iteritems()}   

    n_delays_df= stored_data.loc[stored_data.Indicator=="normalized_delay_responses", col_months]
    n_delays_df = n_delays_df.dropna(axis=1)
    n_delays = {col : value.values[0] for col, value in n_delays_df.iteritems()}   

    months = list(n_total_messag.keys())

    # list of subindicaators dicts
    formula_sub_indicators = [n_total_messag, n_mean_received, n_mean_sent, n_delays]
    # create mapping of month to list of each subindicator value for that month
    aggregated_indicator_formula = {}

    for m in months:
        for d in formula_sub_indicators:
            if m not in aggregated_indicator_formula:
                aggregated_indicator_formula[m]=[d[m]]
            
            elif m not in list(d.keys()):
                continue

            else:
                aggregated_indicator_formula[m].append(d[m])
    
    print(aggregated_indicator_formula)
    # Apply Messages_intensity macroindicator formula (see the description block at the beginning of the notebook)
    formula_output = [round((sum(l[:-1])-l[-1])/3, 2) for l in aggregated_indicator_formula.values()]
    print("")
    print(formula_output)
    # if computed macroindicator value for a month is lower than zero, set it as zero:
    formula_output_without_negative_values = [el if el > 0 else 0 for el in formula_output]
    print("")
    # list of final macroindicator results per month (11 values for 11 months):
    print(formula_output_without_negative_values)

    stored_data.loc[stored_data.Indicator=="intensity_messages_macro", list(aggregated_indicator_formula.keys())]= formula_output_without_negative_values

    return stored_data

def plt_message_intensity_macro(message_intensity_macro, outdir):
    import plotly.graph_objects as go
    # Add data
   
    month = list(message_intensity_macro.keys())
    scores = list(message_intensity_macro.values())

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Messsages_intesity_score',
                            line=dict(color='royalblue', width=4)))
    # Edit the layout
    fig.update_layout(
        title='<b>Evolution of Messaging Intensity by months<br>Scale (min 0, max 1)</b>',
                    xaxis_title='Month',
                    yaxis_title= "Score")
    fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
    fig.add_annotation(
        text = ("<i>**Calculated as an average of 4 sub-indicators normalized values<i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'
    #     , xshift=-1
    #     , yshift=-5
        , font=dict(size=13, color="grey")
        , align="left")
    
    fig.write_image(outdir  +"\\messaging_intensity_macro.png")
    #fig.show()
# %%
