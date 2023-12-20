import os
from itertools import compress
import pandas as pd
import re
from datetime import date, timedelta, datetime

#change the path if necessary
#os.chdir("path")

##create a new folder using the date
#all subfolders are also created based on 4 indicators

#######################################
# list of data
######################################
from datetime import date
output_folder = "output_" + str(date.today())
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files = os.listdir("cleaned_data_path")

#read pilot data
pilot_data =pd.read_excel("pilot_data.xlsx")
pilot_data =pilot_data.set_index("pilot_index")

pilot_list = pilot_data.pilot.astype(str).to_list()

#create pilot subfolder
for plt in pilot_list: 
    output_subfolder = output_folder + "\\" + "\\pilot"+plt
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

list_indicator = ['asymmetry','formality','diversity','intensity']


for subdir in list_indicator : 
    for plt in pilot_list: 
        output_subfolder = output_folder + "\\pilot"+plt+"\\" + subdir
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

#################################
## stored indicator
##############################
stored_data = pd.read_excel("stored_indicator_template.xlsx")
for plt in pilot_list:
    print("Pilot", plt)
    if not os.path.exists(output_folder +"\\pilot"+plt+"\\stored_indicator_pilot"+plt+".xlsx"):
        stored_data.to_excel(output_folder +"\\pilot"+plt+"\\stored_indicator_pilot"+plt+".xlsx", index=False)


##FUNCTION FOR PLOT

#################
#      PLOT     #
#################
def stored_into_dic(stored_data, select_column):
    import re
    col_months = [re.findall(r'\d{4}-\d{2}',x)[0] for x in stored_data.columns.tolist() if bool(re.search(r'\d{4}-\d{2}',x))]
    monthly_data= stored_data.loc[stored_data.Indicator==select_column, col_months]
    monthly_data = monthly_data.dropna(axis=1)
    monthly_dic = {col : value.values[0] for col, value in monthly_data.iteritems()}
    return monthly_dic
############################################################################

###############################################
# data to be used are directly indicated during calculations
#########################################





########################################################################################################################


from saved_functions import communication_diversity_all_functions as func_diversity
from saved_functions import formality_all_functions as func_formality
from saved_functions import messaging_intensity_all_functions as func_message_intensity
from saved_functions import asymmetry_all_functions as func_asymmetry
from saved_functions import profile_static_functions_all as func_profile


pilot_list = ["7","8","9","10"]
for plt in pilot_list:
    print("Pilot", plt)
    #read data
    stored_data = pd.read_excel(output_folder  +"\\pilot"+plt+ "\\stored_indicator_pilot"+plt+".xlsx")

    ################################################
    #### static indicators from profiles
    #########################################
    profile_data = func_profile.get_profile_data_by_project(pilot_data, pilot=int(plt))

    ### fin static indicator####
    gender_diversity = func_profile.get_gender_diversity(profile_data['gender'])
    stored_data.loc[stored_data.Indicator =="gender_diversity","profile_begin"] = gender_diversity
    func_profile.hplt_gender_diversity(gender_diversity, month="Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")

    network_size = func_profile.get_network_size(profile_data['actual_team_size'])
    stored_data.loc[stored_data.Indicator =="network_size","profile_begin"] = network_size

    citizen_researcher_diversity = func_profile.get_citizen_researcher_diversity(profile_data['c_r_diversity'])
    stored_data.loc[stored_data.Indicator =="citizen_researcher_diversity","profile_begin"] = citizen_researcher_diversity
    func_profile.hplt_citizen_researcher_diversity(citizen_researcher_diversity, month="Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")

    collective_skill_diversity = func_profile.get_collective_skill_diversity(profile_data['collective_skill'])
    stored_data.loc[stored_data.Indicator =="collective_skill","profile_begin"] = collective_skill_diversity

    location_diversity = func_profile.get_location_diversity(profile_data['location'])
    stored_data.loc[stored_data.Indicator =="location_diversity","profile_begin"] = location_diversity

    language_diversity = func_profile.get_language_diversity(profile_data['languages'])
    stored_data.loc[stored_data.Indicator =="language_diversity","profile_begin"] = language_diversity

    discipline_diversity = func_profile.get_disciplines_diversity(profile_data['disciplines'])
    stored_data.loc[stored_data.Indicator =="discipline_diversity","profile_begin"] = discipline_diversity

    func_profile.plt_network_diversity_macro(network_size, gender_diversity, citizen_researcher_diversity, month="Dec 2022", pilot = plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")

    func_profile.plt_gender_diversity(gender_diversity, month = "Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    func_profile.plt_network_size(network_size,month = "Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    func_profile.plt_citizen_researcher_diversity(citizen_researcher_diversity, month = "Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    

    #deliverable
    deliverables_diversity  = func_profile.get_deliverables_diversity(profile_data['deliverables'])
    stored_data.loc[stored_data.Indicator =="deliverables_diversity","profile_begin"] = deliverables_diversity[0]

    #funding
    funding_formality = func_profile.get_funding_formality(profile_data['fundings'])
    stored_data.loc[stored_data.Indicator =="funding_formality","profile_begin"] = funding_formality

    # data protection
    data_protection_formality = func_profile.get_data_protection_formality(profile_data['data_protection'])
    stored_data.loc[stored_data.Indicator =="data_protection_formality","profile_begin"] = data_protection_formality

    #recruitment
    recruitment_formality = func_profile.get_recruitment_formality(profile_data['recruitment_data'])
    stored_data.loc[stored_data.Indicator =="recruitment_formality","profile_begin"] = recruitment_formality

    ##contract
    contract_formality = func_profile.get_contract_formality(profile_data['contract_data'])
    stored_data.loc[stored_data.Indicator =="contract_formality","profile_begin"] = contract_formality

    func_profile.hplt_location_diversity(location_diversity, month="Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    func_profile.hplt_language_diversity(language_diversity, month="Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    func_profile.hplt_discipline_diversity(discipline_diversity, month="Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    func_profile.hplt_skill_diversity(collective_skill_diversity, month="Dec 2022",pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")

    func_profile.plt_location_diversity(location_diversity, month="Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    func_profile.plt_language_diversity(language_diversity, month="Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    func_profile.plt_discipline_diversity(discipline_diversity, month="Dec 2022", pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    func_profile.plt_skill_diversity(collective_skill_diversity, month="Dec 2022",pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")

    func_profile.plt_deliverables_diversity(deliverables_diversity, month="Dec 2022",pilot=plt, outdir= output_folder+"\\pilot"+plt+"\\diversity\\")
    func_profile.plt_funding_data_formality(funding_formality, data_protection_formality, month = "Dec 2022", pilot = plt , outdir= output_folder+"\\pilot"+plt+"\\formality\\")

    func_profile.plt_contract_recruitment_formality(contract_formality, recruitment_formality, month = "Dec 2022", pilot = plt, outdir= output_folder+"\\pilot"+plt+"\\formality\\")

    ######################
    ###montly data analysis
    ######################

    #################################
    ## IBM category blogs 
    #######################
    ml = ["2022-06","2022-07","2022-08","2022-09","2022-10","2022-11","2022-12","2023-01","2023-02","2023-03","2023-04","2023-05","2023-06","2023-07","2023-08"]
    declared_dsp = list(profile_data['disciplines'].keys())

    for cm in ml : 
    #check if there was any blog posts

        if os.path.exists("data\\blog\\pilot"+plt+"_"+cm + ".txt"):
            counts_blog = func_asymmetry.get_detected_disciplines_blog(input_data="data\\blog\\pilot"+plt+"_"+cm + ".txt", threshold=0.4, declared_dsp=declared_dsp)
            n_detected_discipline_blog = len(list(counts_blog.keys()))
            found_dsp_blog = list(counts_blog.keys())
            n_blog_counts =func_asymmetry.get_n_counts(counts_blog, k=len(list(profile_data['disciplines'].keys())))
            stored_data.loc[stored_data.Indicator =="discipline_blog_n_found",cm] = n_detected_discipline_blog
            stored_data.loc[stored_data.Indicator =="discpline_blog_found",cm] = "/".join(list(counts_blog.keys()))
            stored_data.loc[stored_data.Indicator =="discpline_blog_declared","profile_begin"] = "/".join(declared_dsp)
            stored_data.loc[stored_data.Indicator =="discipline_deliverables",cm] = n_blog_counts

    check =[("pilot"+plt) in x for x in files]
    pfiles= list(compress(files, check))
    print(pfiles)

    if len(pfiles)==0:
        stored_data.to_excel(output_folder+"\\pilot"+plt+"\\stored_indicator_pilot"+plt+".xlsx", index=False)
        continue


    for f in pfiles:
        print("-------------", f, "--------------------------")
        import pandas as pd
        tmp_d=pd.read_excel("data\\donnees_2023-03-22\\cleaned\\"+f)
        tmp_d.To = tmp_d.To_address
        tmp_d.From = tmp_d.From_address
        tmp_d.to_csv("data\\donnees_2023-03-22\\cleaned\\csv\\"+f.replace("xlsx","csv"))
        
        ##define months        
        
        month_from_filename = re.findall(r'\d{4}-\d{2}',f)[0]
        month_d = datetime.strptime(month_from_filename, "%Y-%m")
        current_month = datetime.strftime(month_d, "%Y-%m")
        previous_month =  datetime.strftime(month_d - timedelta(days =10),"%Y-%m")

        ################################################
        #### diversity : communication
        #########################################
        #the obtained object is a nested dictionary
        input_cleaned_messages_data ="data\\donnees_2023-03-22\\cleaned\\csv\\"+f.replace("xlsx","csv")
        written_language_diversity = func_diversity.get_written_language_diversity(input_cleaned_messages_data)

        print("written_language_diversity",written_language_diversity)

        ##add k from the profile data
        pkl = len(pilot_data.at[int(plt),"languages"].split(","))
        
        
        n_written_language_diveristy = func_diversity.get_n_written_diversity(written_language_diversity, k=pkl)

        print(n_written_language_diveristy)
        stored_data.loc[stored_data.Indicator =="written_language_diversity",month_from_filename] = list(n_written_language_diveristy.values())[0]
        
        language_sophistication = func_diversity.get_language_sophistication(input_cleaned_messages_data)

        print("language_sophistication",language_sophistication)

        ##add k from the profile dataread_data
        pk = pilot_data.loc[int(plt),"actual_team_size"]

        if len(list(language_sophistication[list(language_sophistication.keys())[0]].keys())) > pk: 
            pk = len(list(language_sophistication[list(language_sophistication.keys())[0]].keys()))
    
        n_language_sophistication = func_diversity.get_n_language_sophistication(language_sophistication, k = pk)

        print(n_language_sophistication)

        stored_data.loc[stored_data.Indicator =="language_sophistication",month_from_filename] = list(n_language_sophistication.values())[0]


        ###################################
        # formality of written language
        ###################################

        written_formality = func_formality.get_written_formality(input_cleaned_messages_data)

        #print(written_formality)

        n_written_formality = func_formality.get_n_written_formality(written_formality)

        print("Written language formality")
        print(n_written_formality)
        stored_data.loc[stored_data.Indicator =="formality_written_style",month_from_filename] = list(n_written_formality.values())[0]

        #####################################
        # work formalization and moments of work revision
        ###########################################
        organization, revision = func_formality.get_work_formalisation(input_cleaned_messages_data)


        print("organization")
        print(organization)
        stored_data.loc[stored_data.Indicator =="work_organization",month_from_filename] = list(organization.values())[0]

        print("revision")
        print(revision)
        stored_data.loc[stored_data.Indicator =="work_revision",month_from_filename] = list(revision.values())[0]

        ########################################################################################################################
        ################################################
        #### Intensity messaging
        #########################################

        #####
        # 1. Delays in responses
        #####

        delays  = func_message_intensity.delay_in_email_responses(input = input_cleaned_messages_data)

        print(delays)
        stored_data.loc[stored_data.Indicator =="delay_responses",month_from_filename] = list(delays.values())[0]

        #filtered = {k: v for k, v in delays.items() if v is not None}

        #n_delays = func_message_intensity.get_normalized_delay(delays)

        #print(n_delays)

        #####
        # 2. Mean recieved messages
        #####

        mean_received  = func_message_intensity.get_mean_received(input = input_cleaned_messages_data)

        print(mean_received)

        stored_data.loc[stored_data.Indicator =="mean_number_received_messages",month_from_filename] = list(mean_received.values())[0]

        #n_mean_received = func_message_intensity.get_normalized_mean_received(mean_received)
        #print(n_mean_received)

        #####
        # 3. Mean send messages
        #####
        mean_sent  = func_message_intensity.get_mean_sent(input = input_cleaned_messages_data)

        print(mean_sent)

        stored_data.loc[stored_data.Indicator =="mean_number_sent_messages",month_from_filename] = list(mean_sent.values())[0]

        #n_mean_sent = func_message_intensity.get_normalized_mean_sent(mean_sent)
        #print(n_mean_sent)



        #####
        # 4. Total sent + received messages
        #####

        total_messag = func_message_intensity.get_total_messag(input = input_cleaned_messages_data)

        print(total_messag)
        stored_data.loc[stored_data.Indicator =="total_number_sent_received_messages",month_from_filename] = list(total_messag.values())[0]

        #n_total_messag = func_message_intensity.get_normalized_total_messag(total_messag)
        #print(n_total_messag)


        ################################################
        #### Assymetry
        #########################################

        ####################################
        #stock of knowledge
        ####################################
        data_translated = func_asymmetry.get_translated_data(input_cleaned_messages_data)
        print("data translated")
        data_translated_kept = data_translated.loc[data_translated['From'].isin(profile_data['list_email'].split(",")), : ]
        
        month_to_result = func_asymmetry.get_month_to_result(data_translated_kept)
        print("get month to result")
        #stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",month_from_filename]   = ",".join(list(month_to_result.values())[0])
        if bool(month_to_result) :
            month_to_final_result, monthly_terms=func_asymmetry.get_month_to_final_result_from_store_data(data_translated_kept, stored_data, month_to_result)

            #####PLOT #########
            func_asymmetry.get_wordcloud_fig_by_month(month_to_final_result, outdir =  output_folder+"\\pilot"+plt+"\\asymmetry")
            func_asymmetry.get_wordcloud_fig_by_members(month_to_final_result, outdir =  output_folder+"\\pilot"+plt+"\\asymmetry")
            func_asymmetry.get_pie_plot_by_month(month_to_final_result,outdir =  output_folder+"\\pilot"+plt+"\\asymmetry")

            n_knowledge_asymmetry = func_asymmetry.get_n_knowledge_asymmetry(data_translated_kept, month_to_final_result)
            stored_data.loc[stored_data.Indicator =="stock_knowledge",month_from_filename] = list(n_knowledge_asymmetry.values())[0]

            if pd.isna(stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",previous_month].item()) : 
                stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",month_from_filename]   = ",".join(monthly_terms)

            else:
                stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",month_from_filename]   = stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",previous_month] + ","+",".join(monthly_terms)
        if bool(month_to_result) == False:
            if pd.isna(stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",previous_month].item()) ==False: 
                stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",month_from_filename]   = stored_data.loc[stored_data.Indicator =="stock_knowledge_monthly_words",previous_month] 

        #################################
        ## Discipline messages
        #######################

        #new 20230322
        declared_dsp = list(profile_data['disciplines'].keys())
        counts_message = func_asymmetry.get_detected_disciplines2(data_translated_kept, 0.4, declared_dsp)
        n_detected_discipline = len(list(counts_message.keys()))
        found_dsp = list(counts_message.keys())

        n_message_counts =func_asymmetry.get_n_counts(counts_message, k=len(list(profile_data['disciplines'].keys())))
        stored_data.loc[stored_data.Indicator =="discipline_messages_n_found",month_from_filename] = n_detected_discipline
        stored_data.loc[stored_data.Indicator =="discpline_messages_found",month_from_filename] = "/".join(list(counts_message.keys()))
        stored_data.loc[stored_data.Indicator =="discpline_messages_declared","profile_begin"] = "/".join(declared_dsp)
        stored_data.loc[stored_data.Indicator =="discipline_messages",month_from_filename] = n_message_counts









    # get normalized messages intensity indicator

    func_message_intensity.normalize_from_stored_data(stored_data, "total_number_sent_received_messages","normalized_total_number_sent_received_messages")
    func_message_intensity.normalize_from_stored_data(stored_data, "mean_number_sent_messages","normalized_mean_number_sent_messages")
    func_message_intensity.normalize_from_stored_data(stored_data, "mean_number_received_messages","normalized_mean_number_received_messages")
    func_message_intensity.normalize_from_stored_data_delay(stored_data, "delay_responses","normalized_delay_responses")
    store_data = func_message_intensity.get_messaging_intensity_macro_from_stored_data(stored_data)
    
    ##############
    # PLOT #####
    #############
    
    #################
    # diversity####
    #############
    func_diversity.plt_written_language_diversity_from_stored_data(stored_data,"written_language_diversity",outdir =  output_folder+"\\pilot"+plt+"\\diversity\\")
    func_diversity.plt_language_sophistication_from_stored_data(stored_data=stored_data,select_column ="language_sophistication",outdir =  output_folder+"\\pilot"+plt+"\\diversity\\")
    func_diversity.plt_communication_diversity_from_stored_data(stored_data=stored_data,select_column1 ="written_language_diversity",select_column2 ="language_sophistication",outdir =  output_folder+"\\pilot"+plt+"\\diversity\\")

    ##########
    #formality#
    ###########
    n_written_formality = stored_into_dic(stored_data, "formality_written_style")
    func_formality.plt_written_formality(n_written_formality,outdir =  output_folder+"\\pilot"+plt+"\\formality\\")
    organization = stored_into_dic(stored_data,"work_organization")
    func_formality.plt_organization(organization,outdir =  output_folder+"\\pilot"+plt+"\\formality\\")
    revision= stored_into_dic(stored_data,"work_revision")
    func_formality.plt_revision(revision,outdir =  output_folder+"\\pilot"+plt+"\\formality\\")

    #func_formality.plt_formality_meeting(n_formality_meetings,outdir =  output_folder+"\\formality\\") 
    #func_formality.plt_formality_macro(revision,organization, n_written_formality,n_formality_meetings,outdir =  output_folder+"\\formality\\")

    ##################
    #Intensity#######
    ################
    n_total_messag= stored_into_dic(stored_data,"normalized_total_number_sent_received_messages")
    func_message_intensity.plt_total_messag(n_total_messag,outdir =  output_folder+"\\pilot"+plt+"\\intensity\\")

    n_mean_received= stored_into_dic(stored_data,"normalized_mean_number_received_messages")
    func_message_intensity.plt_mean_received(n_mean_received,outdir =  output_folder+"\\pilot"+plt+"\\intensity\\")

    n_mean_sent= stored_into_dic(stored_data,"normalized_mean_number_sent_messages")
    func_message_intensity.plt_mean_sent(n_mean_sent,outdir =  output_folder+"\\pilot"+plt+"\\intensity\\")

    n_delays= stored_into_dic(stored_data,"normalized_delay_responses")
    func_message_intensity.plt_delays(n_delays,outdir =  output_folder+"\\pilot"+plt+"\\intensity\\")

    messaging_intensity_macro = func_message_intensity.get_messaging_intensity_macro(n_total_messag, n_mean_received, n_mean_sent, n_delays)
    print(messaging_intensity_macro)

    func_message_intensity.plt_message_intensity_macro(messaging_intensity_macro,outdir =  output_folder+"\\pilot"+plt+"\\intensity\\")

    #############
    #Asymmetry###
    ###########
    n_blog_counts= stored_into_dic(stored_data,"discipline_deliverables")
    n_message_counts= stored_into_dic(stored_data,"discipline_messages")

    func_asymmetry.plt_discipline(n_blog_counts, n_message_counts, outdir=output_folder+"\\pilot"+plt+"\\asymmetry\\")


    #export saved result
    stored_data.to_excel(output_folder+"\\pilot"+plt+"\\stored_indicator_pilot"+plt+".xlsx", index=False)


print("all the script was run : END")
