##############################
#get data
########################

def make_dict(data, column):
    d = {x:data[column].split(",").count(x) for x in data[column].split(",")}
    return d


def get_profile_data_by_project(data, pilot=None):
    plt=pilot
    tmp = data.loc[plt]

    profile = {"pilot": plt ,
                "list_email": tmp.list_email,
                    "communication_diversity_k":  tmp.communication_diversity_k,
                    "actual_team_size" :  tmp.actual_team_size,
                    "gender" :  {"female": tmp.female, "male":tmp.male, "other": tmp.gender_other},
                    "c_r_diversity": {"academic":tmp.academic, "activist":tmp.activist, "domain":tmp.domain, "community":tmp.community, "volunteer":tmp.volunteer, "policymaker":tmp.policymaker, "amateur":tmp.amateur, "other":tmp.rc_other},
                    "collective_skill": make_dict(tmp,"collective_skill"),
                    "location": make_dict(tmp,"location"),
                    "languages":make_dict(tmp,"languages"),
                    "disciplines": make_dict(tmp,"disciplines"),
                    "deliverables":   {"Scientific Deliverables" : tmp['Scientific Deliverables'], "Non_scientific Deliverable":  tmp['Non_scientific Deliverable']},
                    "recruitment_data":{"informal": tmp.informal, "call_procedure":tmp.call_procedure, "membership": tmp.membership},
                    "contract_data": {"task limited": tmp['task limited'], "short term":tmp['short term'], "medium term": tmp['medium term'], "long term": tmp['long term']},
                    "number_of_funding_sources": tmp.number_of_funding_sources,
                    "fundings": {"public":tmp.public, "private":tmp.private},
                    "data_protection":{"open_data_repository": tmp.open_data_repository, "access_control": tmp.access_control, "DMP_created" : tmp.DMP_created, "GDPR": tmp.GDPR}
            }
    return profile

#################################
###Shannon entropy formula
##############################
def balance(seq, k=None):
    from collections import Counter
    from numpy import log
    
    n = sum(seq.values())
    classes = [(clas,float(count)) for clas,count in Counter(seq).items()]
    if not k:
        k = len(classes)
    H = -sum([(count/n) * log((count/n)) for clas,count in classes if count!=0])
    return H/log(k)


###############################
#####network size subindicator
##################################
def get_network_size(actual_team_size):
    maximum_team_size = 10
    network_size =  actual_team_size/maximum_team_size
    return network_size

##############################
##gender diversity subindicator
################################

def get_gender_diversity(gender):
    total_gender_options = 3
    return balance(gender, k=total_gender_options)


#####################################
### citizen researcher diveristy
######################################
def get_citizen_researcher_diversity(c_r_diversity):
    total_options = 8
    return balance(c_r_diversity, k=total_options)



###################################
### diversity indicator
##############################
#Diversity of Gender (shows diversity in the Scale from 0 to 1, where 1 means that 3 genders are presented in the team. 3 genders is the total number of genders to select for users)
def hplt_gender_diversity(gender_diversity, month ="selected month", pilot=None, outdir = None):  
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
                x=[gender_diversity],
                y=['Gender diversity'],
                orientation='h', marker_color='skyblue', textposition='auto'))
    # Edit the layout
    fig.update_layout(#title='<b>Change of Collective  Backgound Diversity Sub-indicator</b><br>\
    #<i>Description... <i><br>
    )
    fig.update_layout(xaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.27
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title=f" Gender diveristy for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )

    #fig.show()     
    fig.write_image(outdir+f"h_gender_diveristy_for_{month}_Pilot_{pilot}"+".png")

def plt_gender_diversity(gender_diversity, month="Insert Month", pilot = None, outdir = None):
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = gender_diversity,                                                       # actual normalized value
        mode = "gauge+number",
        title = {'text': "Gender diversity"},
            gauge = {'axis': {'range': [None, 1]},                           # 1 is a max value
                'bar': {'color': "salmon"},
                'bgcolor': "white"}))

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    #fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Gender diveristy for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )

    #fig.show()
    fig.write_image(outdir+f"gender_diveristy_for_{month}_Pilot_{pilot}"+".png")


#############################################
### Network size
#####################################
def plt_network_size(network_size, month="Insert Month", pilot = None, outdir = None):
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = network_size,                                                       # actual normalized value 
    mode = "gauge+number",
    title = {'text': "Network size"},
    gauge = {'axis': {'range': [None, 1]},'bar': {'color': "MidnightBlue"},
                 'steps' : [
                 {'range': [0, 0.2], 'color': "#b3ecec"},
                 {'range': [0.2, 0.4], 'color': "#89ecda"},
                 {'range': [0.4, 0.6], 'color': "Turquoise"},
                 {'range': [0.6, 0.8], 'color': "MediumTurquoise"},
                 {'range': [0.8, 1], 'color': "DarkTurquoise"}]}                          #1 is a max value in diversity
    ))
    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'
        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Network size for {month} / Pilot  {pilot}")

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    #fig.show()
    fig.write_image(outdir+"network_size_"+f"{month}_{pilot}"+".png")

###############################
## citizen researcher diversity
#################################
def hplt_citizen_researcher_diversity(citizen_researcher_diversity, month="Insert Month", pilot = None,outdir=None): 
    #Diversity of Citizen/Researcher (shows diversity in the Scale from 0 to 1, where 1 means that 8 Citizen/Researcher categories are presented in the team. 8 Citizen/Researcher categories is the total number of  Citizen/Researcher categories to select for users)
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
                x=[citizen_researcher_diversity],
                y=['Citizen/Researcher diversity'],
                orientation='h', marker_color='lightpink', textposition='auto'))
    # Edit the layout
    fig.update_layout(#title='<b>Change of Collective  Backgound Diversity Sub-indicator</b><br>\
    #<i>Description... <i><br>
    )
    fig.update_layout(xaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.27
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title=f" Citizen/Researcher diveristy for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )

    #fig.show()
    fig.write_image(outdir+f"h_Citizen_Researcher_diveristy_for_{month}_Pilot_{pilot}"+".png")

def plt_citizen_researcher_diversity(citizen_researcher_diversity, month="Insert Month", pilot = None, outdir=None): 

    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = 0.27,                                                       # actual normalized value
        mode = "gauge+number",
        title = {'text': "Citizen/Researcher diversity"},
        gauge = {'axis': {'range': [None, 1]},                             # 1 is a max value
                'bar': {'color': "darkblue"},
                'bgcolor': "white"}))

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Citizen/Researcher diveristy for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )

    #fig.show() 
    fig.write_image(outdir+f"Citizen_Researcher_diveristy_for_{month}_Pilot_{pilot}"+".png")

######################################
## Network diversity macroindicator 
######################################
def plt_network_diversity_macro(network_size, gender_diversity, citizen_researcher_diversity, month="Insert Month", pilot = None, outdir=None):
    import plotly.graph_objects as go

    months = [month]  #add here a new month if a new member joins/leaves the team

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months,
        y=[gender_diversity],
        name='Gender diversity',
        marker_color='skyblue', textposition='auto'
    ))
    fig.add_trace(go.Bar(
        x=months,
        y=[network_size],
        name='Network size',
        marker_color='gainsboro', textposition='auto'
    ))
    fig.add_trace(go.Bar(
        x=months,
        y=[citizen_researcher_diversity],
        name='Citizen/Researcher diversity',
        marker_color='lightpink', textposition='auto'
    ))


    # Edit the layout
    fig.update_layout(#title='<b>Change of Network Diversity Sub-indicator</b><br>\
    #<i>Description... <i><br>',
                    xaxis_title='Month',
                    yaxis_title='Score')
    fig.update_layout(yaxis_range=[0,1])




    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title= f"{month} / Pilot  {pilot}")
    #fig.show()
    fig.write_image(outdir+"network_diversity_macro_"+f"{month}_{pilot}"+".png")


    ################################################################
    ##profile diversity
    #######################################################

#######################
#Collective skill
##########################
def get_collective_skill_diversity(collective_skill):
    total_number_of_skills  = 25  
    return balance(collective_skill, k=total_number_of_skills)

def get_location_diversity(location):
    total_number_of_locations = 62
    return balance(location, total_number_of_locations)

def get_language_diversity(languages): 
    total_number_of_languages = 24
    return balance(languages, total_number_of_languages)

def get_disciplines_diversity(disciplines):
    total_number_of_disciplines = 28
    return balance(disciplines, total_number_of_disciplines)


############################
# Plot profile diversity
############################

def hplt_location_diversity(location, month="Insert Month", pilot=None, outdir=None):
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
                x=[location],
                y=['Location'], 
                orientation='h', marker_color='indianred', textposition='auto'))
    # Edit the layout
    fig.update_layout(#title='<b>Change of Collective  Backgound Diversity Sub-indicator</b><br>\                     # here change the title of plot
    #<i>Description... <i><br>
    )
    fig.update_layout(xaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text =("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Location diveristy for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )

    #fig.show()
    fig.write_image(outdir+f"h_location_diveristy_{month}_Pilot_{pilot}"+".png")

def hplt_language_diversity(language_diversity, month="Insert Month", pilot=None, outdir = None):
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
                x=[language_diversity],
                y=['Languages'],
                orientation='h', marker_color='lightsalmon', textposition='auto'))
    # Edit the layout
    fig.update_layout(#title='<b>Change of Collective  Backgound Diversity Sub-indicator</b><br>\
    #<i>Description... <i><br>
    )
    fig.update_layout(xaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Language diveristy for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )


    #fig.show()
    fig.write_image(outdir+f"h_language_diversity_{month}_Pilot_{pilot}"+".png")


def hplt_discipline_diversity(discipline_diveristy, month="Insert Month", pilot=None, outdir=None):
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
                x=[discipline_diveristy],
                y=['Disciplines'],
                orientation='h', marker_color='lightblue', textposition='auto'))
    # Edit the layout
    fig.update_layout(#title='<b>Change of Collective  Backgound Diversity Sub-indicator</b><br>\
    #<i>Description... <i><br>
    )
    fig.update_layout(xaxis_range=[0,1])
    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Discipline diveristy for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )



    #fig.show()
    fig.write_image(outdir+f"h_discipline_diversity_{month}_Pilot_{pilot}"+".png")


def hplt_skill_diversity(collective_skill, month="Insert Month", pilot=None, outdir=None):
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
                x=[collective_skill],
                y=['Skills'],
                orientation='h', marker_color='mediumpurple', textposition='auto'))
    # Edit the layout
    fig.update_layout(#title='<b>Change of Collective  Backgound Diversity Sub-indicator</b><br>\
    #<i>Description... <i><br>
    )
    fig.update_layout(xaxis_range=[0,1])


    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Collective skill diveristy for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )

    #fig.show()
    fig.write_image(outdir+f"h_collective_skill_diversity_{month}_Pilot_{pilot}"+".png")



def plt_skill_diversity(collective_skill, month="Insert Month", pilot=None, outdir = None):
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = collective_skill,                                                       # actual normalized value 
        mode = "gauge+number",
        title = {'text': "Skills"},
        gauge = {'axis': {'range': [None, 1]}, 'bgcolor': "aliceblue", 'bar': {'color': "black"},
                'steps' : [
                    {'range': [0, 0.2], 'color': "#5ced73"},
                    {'range': [0.2, 0.4], 'color': "#39e75f"},
                    {'range': [0.4, 0.6], 'color': "#1fd655"},
                    {'range': [0.6, 0.8], 'color': "#00c04b"},
                    {'range': [0.8, 1], 'color': "#00ab41"}]}                          #1 is a max value in diversity
    ))
    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'
        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Collective skill diveristy for {month} / Pilot  {pilot}")

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    #fig.show()
    fig.write_image(outdir+f"collective_skill_diversity_{month}_Pilot_{pilot}"+".png")


def plt_discipline_diversity(discipline_diveristy, month="Insert Month", pilot=None, outdir = None):
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = discipline_diveristy,                                                       # actual normalized value 
        mode = "gauge+number",
        title = {'text': "Disciplines"},
        gauge = {'axis': {'range': [None, 1]},'bar': {'color': "black"},
                'steps' : [
                    {'range': [0, 0.2], 'color': "orange"},
                    {'range': [0.2, 0.4], 'color': "coral"},
                    {'range': [0.4, 0.6], 'color': "orangered"},
                    {'range': [0.6, 0.8], 'color': "red"},
                    {'range': [0.8, 1], 'color': "crimson"}]}                          #1 is a max value in diversity
    ))
    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = high diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'
        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Discipline diveristy for {month} / Pilot  {pilot}")

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    #fig.show()
    fig.write_image(outdir+f"discipline_diversity_{month}_Pilot_{pilot}"+".png")


def plt_language_diversity(language_diversity, month="Insert Month", pilot=None, outdir = None):
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = language_diversity,                                                       # actual normalized value 
        mode = "gauge+number",
        title = {'text': "Languages"},
        gauge = {'axis': {'range': [None, 1]},'bar': {'color': "black"},
                'steps' : [
                    {'range': [0, 0.2], 'color': "#c2b8d1"},
                    {'range': [0.2, 0.4], 'color': "mediumpurple"},
                    {'range': [0.4, 0.6], 'color': "blueviolet"},
                    {'range': [0.6, 0.8], 'color': "darkviolet"},
                    {'range': [0.8, 1], 'color': "rebeccapurple"}]}                          #1 is a max value in diversity
    ))
    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = hight diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'
        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Language diveristy for {month} / Pilot  {pilot}")

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    #fig.show()
    fig.write_image(outdir+f"language_diversity_{month}_Pilot_{pilot}"+".png")


def plt_location_diversity(location, month="Insert Month", pilot=None, outdir = None):
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = location,                                                       # actual normalized value 
        mode = "gauge+number",
        title = {'text': "Location"},
        gauge = {'axis': {'range': [None, 1]},'bar': {'color': "black"},
                'steps' : [
                    {'range': [0, 0.2], 'color': "#e5a0c6"},
                    {'range': [0.2, 0.4], 'color': "#e27bb1"},
                    {'range': [0.4, 0.6], 'color': "#e2619f"},
                    {'range': [0.6, 0.8], 'color': "#e44b8d"},
                    {'range': [0.8, 1], 'color': "#d24787"}]}                          #1 is a max value in diversity
    ))
    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low diversity <br> 1 = hight diversity <i>")
        , showarrow=False
        , x = 0
        , y = -0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'
        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Location diveristy for {month} / Pilot  {pilot}")

    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    #fig.show()
    fig.write_image(outdir+f"location_diversity_{month}_Pilot_{pilot}"+".png")


#############################
# target deliverables
###########################

def get_deliverables_diversity(deliverables):

    total = sum(deliverables.values(), 0.0)
    new_deliverables = {k: v / total*100 for k, v in deliverables.items()}
    return list(new_deliverables.values())

def plt_deliverables_diversity(deliverables, month="Insert Month",pilot=None, outdir = None):
    import plotly.graph_objects as go
    colors = ['lightgreen', 'gold']
    labels=['Scietific Deliverable','Non-scientific Deliverable']
    values=deliverables

    fig = go.Figure(data=[go.Pie(labels=labels,
                                values=values,  insidetextorientation='auto')])
    fig.update_traces(textinfo='percent', textfont_size=40,
                    marker=dict(colors=colors, line=dict(color='#000000', width=3)))
    fig.update_layout(title=f" Deliverable diveristy for {month} / Pilot  {pilot}")

    #fig.show()
    fig.write_image(outdir+f"deliverable_diversity_{month}_Pilot_{pilot}"+".png")

##########################
###funding formality
########################


def get_funding_formality(funding):
    public_source_weight = 2
    private_source_weight  = 1
    number_of_funding_sources = sum(funding.values())
    for k, v in funding.items():
       if k == "public":
        distribution_1 = v * public_source_weight
       else:
        distribution_2 = v * private_source_weight
    result = distribution_1 + distribution_2
    max = number_of_funding_sources *  public_source_weight    #1*2 one public
    min = number_of_funding_sources *  private_source_weight    #1*1 one private

    normalized = (result - min)/(max-min)
    return normalized

###################
##data compliance
########################

def get_data_protection_formality(data_protection):
    number_of_questions = 4
    yes_weight = 2
    no_weight  = 1
    yes_count = 0
    no_count = 0
    for key in data_protection:
        if data_protection[key] == "yes":
            yes_count = yes_count + 1
        else:
            no_count = no_count + 1
    result = (yes_count * yes_weight) + (no_count * no_weight)
    max = number_of_questions * yes_weight    # 4 yes answers
    min = number_of_questions * no_weight    # 4 no answers

    normalized = (result - min)/(max-min)

    return normalized


#########################################
#plot formality subindicators
################################
def plt_funding_data_formality(funding_formality, data_protection_formality, month = "Insert Month", pilot = None , outdir=None):
    import plotly.graph_objects as go

    months = [month]  #add here a new month if a new member joins/leaves the team

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months,
        y=[funding_formality],
        name='Distribution of funding sources obtained',
        marker_color='blue', textposition='auto', text = '<b>1<b>', # add a value above the bars, make it bold
        textfont=dict(
            family='sans serif',
            size=11, # font of value above the bars

        )
    ))
    fig.add_trace(go.Bar(
        x=months,
        y=[data_protection_formality],
        name='Data protection compliance',
        marker_color='lightgreen', textposition='auto', text = '<b>1<b>',
        textfont=dict(
            family='sans serif',
            size=11,

        )
    ))

    # Edit the layout
    fig.update_layout(#title='<b>Change of Formality Sub-indicator</b><br>\
    #<i>Description... <i><br>',
                    yaxis_title='Formality')
    fig.update_layout(yaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low formality  <br> 1 = high formality <i>")
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
    fig.update_layout(title=f" Funding/Data protection formality for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.update_traces(marker_line_color = 'black', marker_line_width = 1) # a line around the bars and legend marks

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=360)
    #fig.show()
    fig.write_image(outdir+f"funding_data_protection_formality_{month}_Pilot_{pilot}"+".png")


#######################
# Recruitment formality
#######################
def get_recruitment_formality(recruitment_data):
    number_of_users = sum(recruitment_data.values())
    informal_type_weight = 1
    membership_type_weight = 2
    call_procedure_type_weight = 3
    for k, v in recruitment_data.items():
        if k == "informal":
            distribution_1 = v * informal_type_weight
        elif k == "membership":
            distribution_2 = v * membership_type_weight
        else:
            distribution_3 = v * call_procedure_type_weight
    result = distribution_1 + distribution_2 + distribution_3
    max_recr = number_of_users *  call_procedure_type_weight    #4*3 = 12  all  4 members are recruited by the most formal call_procedure_type
    min_recr = number_of_users *  informal_type_weight    #4*1 = 4 all  4 members are recruited by the the the most informal informal_type_weight
    normalized = (result- min_recr)/(max_recr-min_recr)
    return normalized

##########################
##contract formality
###########################
def get_contract_formality(contract_data):
    number_of_users = sum(contract_data.values())
    task_limited_weight = 4 
    short_term_weight = 3
    medium_term_weight = 2
    long_term_weight = 1
    for k, v in contract_data.items():
        if k == "task limited":
            dist_1 = v * task_limited_weight
        elif k == "short term":
            dist_2 = v * short_term_weight
        elif k == "medium term":
            dist_3 = v * short_term_weight
        else:
            dist_4 = v * long_term_weight
    result =  dist_1 + dist_2 + dist_3 + dist_4
    max_contr = number_of_users * task_limited_weight  # 4* 4 = 16 all  4 members are recruited by the most formal contract time
    min_contr = number_of_users * long_term_weight    #4*1 = 4 all  4 members are recruited by the the the most informal 
    normalized = (result- min_contr)/(max_contr-min_contr)
    return normalized

def plt_contract_recruitment_formality(contract_formality, recruitment_formality, month = "Insert Month", pilot = None, outdir=None ):
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
                x=[contract_formality, recruitment_formality],
                y=['Staff contract duration', 'Recruitment mode'],
                orientation='h', marker_color='lightskyblue', textposition='inside'))

    # Edit the layout
    fig.update_layout(#title='<b>Change of Collective  Backgound Diversity Sub-indicator</b><br>\
    #<i>Description... <i><br>
    )
    fig.update_layout(xaxis_range=[0,1])

    # add a footnote to the bottom left 
    fig.add_annotation(
        text = ("<i>Scale: <br> 0 = low formality  <br> 1 = high formality <i>")
        , showarrow=False
        , x = 0
        , y = -0.27
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'

        , font=dict(size=16, color="black")
        , align="left")
    fig.update_layout(margin=dict(t=150))
    fig.update_layout(title={'font': {'size': 20}})
    fig.update_layout(title=f" Contract / Recruitment formality for {month} / Pilot  {pilot}")


    layout = go.Layout(
        autosize=False,
        width=2000,
        height=2000,
    )
    fig.update_traces(marker_line_color = 'black', marker_line_width = 1) # a line around the bars and legend marks

    #fig.show()
    fig.write_image(outdir+f"contract_recruitment_formality_{month}_Pilot_{pilot}"+".png")


