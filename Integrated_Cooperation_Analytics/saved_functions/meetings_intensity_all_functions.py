#!/usr/bin/env python
# coding: utf-8

##################################
# data extraction and cleaning
##################################

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


###################################
# 1. Total number of meetings 
###################################

def get_total_n_of_meetings(input_data):
    import calendar 
    data = read_meetings_data(input_data)
    available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
    dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]
    total_n_of_meetings = dict()
    for i, df in zip(available_months, dfs):
        try:
            index = df.index
            no_of_meetings = len(index)
            #print(result)
            total_n_of_meetings.update({calendar.month_name[i] : no_of_meetings})
        except KeyError:
            print(f"key error for month {i}")
            continue
    return total_n_of_meetings


###################################
# 2. Mean duration of meetings 
###################################
def get_mean_duration(input_data):
    import calendar
    import numpy as np
    import statistics
    import pandas as pd

    data = read_meetings_data(input_data)
    #create new column with Duration
    data['End Time'] = [x + ':00' for x in data['End Time']]
    data['Start Time'] = [y + ':00' for y in data['Start Time']]
    data['DURATION'] = (data['End Time']).astype('datetime64[ns]') - (data['Start Time']).astype('datetime64[ns]')
    available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
    dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]
    mean_duration = dict()
    for i, df in zip(available_months, dfs):
        try:
            converted = (pd.to_timedelta(df['DURATION']) / np.timedelta64(1, 'm')).astype(int)
            mean_value = statistics.mean(converted)
            #convert Timedelta to numeric value in order to get the mean value
            mean_duration.update({calendar.month_name[i] : round(mean_value)})
        except KeyError:
            print(f"key error for month {i}")
            continue
    return mean_duration

###################################
# 3. invitees scale and attendance rate 
###################################
def get_invitees_scale_attendance_rate(input_data):
    import calendar
    import re 

    data = read_meetings_data(input_data)
    #create new column with Duration
    available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
    dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]
    invitee_scale = dict()
    attendance_rate = dict()

    for i, df in zip(available_months, dfs):
        try:
            values = []
            for value in df['Guests'].values:
                values.append(value)
            regex = 'email'        #find the number of all mentionning of emails in Guests column     
            matches = re.findall(regex, str(values)) 
            meeting_invites = (len(matches)) 
            invitee_scale.update({calendar.month_name[i] :meeting_invites})
            
            regex2 = 'accepted'         #find the number of all person whose responseStatus is "accepted"     
            matches = re.findall(regex2, str(values)) 
            invite_accepted = (len(matches)) 
            meeting_attandance =  (invite_accepted * 100)/meeting_invites #calculated as the real participants percentage value of the total number of invited ones. 
            attendance_rate.update({calendar.month_name[i] :meeting_attandance})
        except KeyError:
            print(f"key error for month {i}")
            continue
    return invitee_scale, attendance_rate

#########################################
### Normalization 
##################################

def normalize_with_rolling_max(months, values):
    max_v = max(values)
    normalized = [(round(value/max_v, 2)) for value in values]
    d = {k:v for k, v in zip(months,normalized)}
    return d

def get_normalized_total_n_of_meetings(total_n_of_meetings):
    months = list(total_n_of_meetings.keys())
    values = list(total_n_of_meetings.values())
    total_n_of_meetings = normalize_with_rolling_max(months, values)
    return total_n_of_meetings

def get_normalized_mean_duration(mean_duration):
    months = list(mean_duration.keys())
    values = list(mean_duration.values())
    mean_duration = normalize_with_rolling_max(months, values)
    return mean_duration

def get_normalized_invitee_scale(invitee_scale):
    months = list(invitee_scale.keys())
    values = list(invitee_scale.values())
    invitee_scale = normalize_with_rolling_max(months, values)
    return invitee_scale

def get_normalized_attendance(attendance):
    months = list(attendance.keys())
    values = list(attendance.values())
    attendance = normalize_with_rolling_max(months, values)
    return attendance

################################
## Meeting Intensity Macroindicator 
######################################
def get_meeting_intensity_macro(n_total_n_of_meetings, n_attendance, n_invitee_scale, n_mean_duration):
    import numpy as np
    #get months as keys of any of subindicators
    months = list(n_total_n_of_meetings.keys())

    aggregated_indicator= {}
    sub_indicators = [n_total_n_of_meetings, n_attendance, n_invitee_scale, n_mean_duration]
    print(sub_indicators)
    print("")
    for m in months:
        for d in sub_indicators:
            if m not in aggregated_indicator:
                aggregated_indicator[m]=[d[m]]
            else:
                aggregated_indicator[m].append(d[m])
    aggregated_indicator_average = {k:round(np.mean(np.array(v)), 2) for k,v in aggregated_indicator.items()}  

    print(aggregated_indicator_average)
    return  aggregated_indicator_average

########################
###plot
############################
def plt_meeting_intensity_macro(meeting_intensity_macro, outdir):
    import plotly.graph_objects as go
    # Add data
   
    month = list(meeting_intensity_macro.keys())
    scores = list(meeting_intensity_macro.values())

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=month, y=scores, name='Meeting_intesity_score',
                            line=dict(color='royalblue', width=4)))
    # Edit the layout
    fig.update_layout(
        title='<b>Evolution of Meetings Intensity by months <br>Scale (min 0, max 1)<br> </b>',
                    xaxis_title='Month',
                    yaxis_title='Score')
    
    fig.write_image(outdir  +"\\meeting_intensity_macro.png")
    #fig.show()