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

def read_data(input_data=None):
    import pandas as pd
    from datetime import datetime
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
    
    # convert Date_Converted to datetime in order to sort df by months
    data["Date_Converted_dt"] = pd.to_datetime(data.Date_Converted, utc=True)

    data = data.sort_values('Date_Converted_dt', ascending=True) ###sort all data by Date_converted values

    return data


#######################################################
## written language diversity
###################################################
#clean dataset, no additional preprocessing is needed
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
        new6 = new5.replace("\n","").replace("\r","")
        sentences.append(new6)
    return sentences


def detect_languages(clean_values):
    from langdetect import detect
    #list of languages 
    all_languages_codes =   {
  	'ab': 'Abkhazian',
  	'abk': 'Abkhazian',
  	'aa': 'Afar',
  	'aar': 'Afar',
  	'af': 'Afrikaans',
  	'afr': 'Afrikaans',
  	'sq': 'Albanian',
  	'alb/sqi*': 'Albanian',
  	'am': 'Amharic',
  	'amh': 'Amharic',
  	'ar': 'Arabic',
  	'ara': 'Arabic',
  	'an': 'Aragonese',
  	'arg': 'Aragonese',  	'ab': 'Abkhazian',
  	'abk': 'Abkhazian',
  	'aa': 'Afar',
  	'aar': 'Afar',
  	'af': 'Afrikaans',
  	'afr': 'Afrikaans',
  	'sq': 'Albanian',
  	'alb/sqi*': 'Albanian',
  	'am': 'Amharic',
  	'amh': 'Amharic',
  	'ar': 'Arabic',
  	'ara': 'Arabic',
  	'an': 'Aragonese',
  	'arg': 'Aragonese',
  	'hy': 'Armenian',
  	'arm/hye*': 'Armenian',
  	'as': 'Assamese',
  	'hy': 'Armenian',
  	'arm/hye*': 'Armenian',
  	'as': 'Assamese',
  	'asm': 'Assamese',
  	'ae': 'Avestan',
  	'ave': 'Avestan',
  	'ay': 'Aymara',
  	'aym': 'Aymara',
  	'az': 'Azerbaijani',
  	'aze': 'Azerbaijani',
  	'ba': 'Bashkir',
  	'bak': 'Bashkir',
  	'eu': 'Basque',
  	'baq/eus*': 'Basque',
  	'be': 'Belarusian',
  	'bel': 'Belarusian',
  	'bn': 'Bengali',
  	'ben': 'Bengali',
  	'bh': 'Bihari',
  	'bih': 'Bihari',
  	'bi': 'Bislama',
  	'bis': 'Bislama',
  	'bs': 'Bosnian',
  	'bos': 'Bosnian',
  	'br': 'Breton',
  	'bre': 'Breton',
  	'bg': 'Bulgarian',
  	'bul': 'Bulgarian',
  	'my': 'Burmese',
  	'bur/mya*': 'Burmese',
  	'ca': 'Catalan',
  	'cat': 'Catalan',
  	'ch': 'Chamorro',
  	'cha': 'Chamorro',
  	'ce': 'Chechen',
  	'che': 'Chechen',
  	'zh': 'Chinese',
  	'zh-cn': 'Chinese (Simplified)',
  	'zh-tw': 'Chinese (Traditional)',
  	'chi/zho*': 'Chinese',
  	'cu': 'Church Slavic; Slavonic; Old Bulgarian',
  	'chu': 'Church Slavic; Slavonic; Old Bulgarian',
  	'cv': 'Chuvash',
  	'chv': 'Chuvash',
  	'kw': 'Cornish',
  	'cor': 'Cornish',
  	'co': 'Corsican',
  	'cos': 'Corsican',
  	'hr': 'Croatian',
  	'scr/hrv*': 'Croatian',
  	'cs': 'Czech',
  	'cze/ces*': 'Czech',
  	'da': 'Danish',
  	'dan': 'Danish',
  	'dv': 'Divehi; Dhivehi; Maldivian',
  	'div': 'Divehi; Dhivehi; Maldivian',
  	'nl': 'Dutch',
  	'dut/nld*': 'Dutch',
  	'dz': 'Dzongkha',
  	'dzo': 'Dzongkha',
  	'en': 'English',
  	'eng': 'English',
  	'eo': 'Esperanto',
  	'epo': 'Esperanto',
  	'et': 'Estonian',
  	'est': 'Estonian',
  	'fo': 'Faroese',
  	'fao': 'Faroese',
  	'fj': 'Fijian',
  	'fij': 'Fijian',
  	'fi': 'Finnish',
  	'fin': 'Finnish',
  	'fr': 'French',
  	'fre/fra*': 'French',
  	'gd': 'Gaelic; Scottish Gaelic',
  	'gla': 'Gaelic; Scottish Gaelic',
  	'gl': 'Galician',
  	'glg': 'Galician',
  	'ka': 'Georgian',
  	'geo/kat*': 'Georgian',
  	'de': 'German',
  	'ger/deu*': 'German',
  	'el': 'Greek, Modern (1453-)',
  	'gre/ell*': 'Greek, Modern (1453-)',
  	'gn': 'Guarani',
  	'grn': 'Guarani',
  	'gu': 'Gujarati',
  	'guj': 'Gujarati',
  	'ht': 'Haitian; Haitian Creole',
  	'hat': 'Haitian; Haitian Creole',
  	'ha': 'Hausa',
  	'hau': 'Hausa',
  	'he': 'Hebrew',
  	'heb': 'Hebrew',
  	'hz': 'Herero',
  	'her': 'Herero',
  	'hi': 'Hindi',
  	'hin': 'Hindi',
  	'ho': 'Hiri Motu',
  	'hmo': 'Hiri Motu',
  	'hu': 'Hungarian',
  	'hun': 'Hungarian',
  	'is': 'Icelandic',
  	'ice/isl*': 'Icelandic',
  	'io': 'Ido',
  	'ido': 'Ido',
  	'id': 'Indonesian',
  	'ind': 'Indonesian',
  	'ia': 'Interlingua (International Auxiliary Language Association)',
  	'ina': 'Interlingua (International Auxiliary Language Association)',
  	'ie': 'Interlingue',
  	'ile': 'Interlingue',
  	'iu': 'Inuktitut',
  	'iku': 'Inuktitut',
  	'ik': 'Inupiaq',
  	'ipk': 'Inupiaq',
  	'ga': 'Irish',
  	'gle': 'Irish',
  	'it': 'Italian',
  	'ita': 'Italian',
  	'ja': 'Japanese',
  	'jpn': 'Japanese',
  	'jv': 'Javanese',
  	'jav': 'Javanese',
  	'kl': 'Kalaallisut',
  	'kal': 'Kalaallisut',
  	'kn': 'Kannada',
  	'kan': 'Kannada',
  	'ks': 'Kashmiri',
  	'kas': 'Kashmiri',
  	'kk': 'Kazakh',
  	'kaz': 'Kazakh',
  	'km': 'Khmer',
  	'khm': 'Khmer',
  	'ki': 'Kikuyu; Gikuyu',
  	'kik': 'Kikuyu; Gikuyu',
  	'rw': 'Kinyarwanda',
  	'kin': 'Kinyarwanda',
  	'ky': 'Kirghiz',
  	'kir': 'Kirghiz',
  	'kv': 'Komi',
  	'kom': 'Komi',
  	'ko': 'Korean',
  	'kor': 'Korean',
  	'kj': 'Kuanyama; Kwanyama',
  	'kua': 'Kuanyama; Kwanyama',
  	'ku': 'Kurdish',
  	'kur': 'Kurdish',
  	'lo': 'Lao',
  	'lao': 'Lao',
  	'la': 'Latin',
  	'lat': 'Latin',
  	'lv': 'Latvian',
  	'lav': 'Latvian',
  	'li': 'Limburgan; Limburger; Limburgish',
  	'lim': 'Limburgan; Limburger; Limburgish',
  	'ln': 'Lingala',
  	'lin': 'Lingala',
  	'lt': 'Lithuanian',
  	'lit': 'Lithuanian',
  	'lb': 'Luxembourgish; Letzeburgesch',
  	'ltz': 'Luxembourgish; Letzeburgesch',
  	'mk': 'Macedonian',
  	'mac/mkd*': 'Macedonian',
  	'mg': 'Malagasy',
  	'mlg': 'Malagasy',
  	'ms': 'Malay',
  	'may/msa*': 'Malay',
  	'ml': 'Malayalam',
  	'mal': 'Malayalam',
  	'mt': 'Maltese',
  	'mlt': 'Maltese',
  	'gv': 'Manx',
  	'glv': 'Manx',
  	'mi': 'Maori',
  	'mao/mri*': 'Maori',
  	'mr': 'Marathi',
  	'mar': 'Marathi',
  	'mh': 'Marshallese',
  	'mah': 'Marshallese',
  	'mo': 'Moldavian',
  	'mol': 'Moldavian',
  	'mn': 'Mongolian',
  	'mon': 'Mongolian',
  	'na': 'Nauru',
  	'nau': 'Nauru',
  	'nv': 'Navaho, Navajo',
  	'nav': 'Navaho, Navajo',
  	'nd': 'Ndebele, North',
  	'nde': 'Ndebele, North',
  	'nr': 'Ndebele, South',
  	'nbl': 'Ndebele, South',
  	'ng': 'Ndonga',
  	'ndo': 'Ndonga',
  	'ne': 'Nepali',
  	'nep': 'Nepali',
  	'se': 'Northern Sami',
  	'sme': 'Northern Sami',
  	'no': 'Norwegian',
  	'nor': 'Norwegian',
  	'nb': 'Norwegian Bokmal',
  	'nob': 'Norwegian Bokmal',
  	'nn': 'Norwegian Nynorsk',
  	'nno': 'Norwegian Nynorsk',
  	'ny': 'Nyanja; Chichewa; Chewa',
  	'nya': 'Nyanja; Chichewa; Chewa',
  	'oc': 'Occitan (post 1500); Provencal',
  	'oci': 'Occitan (post 1500); Provencal',
  	'or': 'Oriya',
  	'ori': 'Oriya',
  	'om': 'Oromo',
  	'orm': 'Oromo',
  	'os': 'Ossetian; Ossetic',
  	'oss': 'Ossetian; Ossetic',
  	'pi': 'Pali',
  	'pli': 'Pali',
  	'pa': 'Panjabi',
  	'pan': 'Panjabi',
  	'fa': 'Persian',
  	'per/fas*': 'Persian',
  	'pl': 'Polish',
  	'pol': 'Polish',
  	'pt': 'Portuguese',
  	'por': 'Portuguese',
  	'ps': 'Pushto',
  	'pus': 'Pushto',
  	'qu': 'Quechua',
  	'que': 'Quechua',
  	'rm': 'Raeto-Romance',
  	'roh': 'Raeto-Romance',
  	'ro': 'Romanian',
  	'rum/ron*': 'Romanian',
  	'rn': 'Rundi',
  	'run': 'Rundi',
  	'ru': 'Russian',
  	'rus': 'Russian',
  	'sm': 'Samoan',
  	'smo': 'Samoan',
  	'sg': 'Sango',
  	'sag': 'Sango',
  	'sa': 'Sanskrit',
  	'san': 'Sanskrit',
  	'sc': 'Sardinian',
  	'srd': 'Sardinian',
  	'sr': 'Serbian',
  	'scc/srp*': 'Serbian',
  	'sn': 'Shona',
  	'sna': 'Shona',
  	'ii': 'Sichuan Yi',
  	'iii': 'Sichuan Yi',
  	'sd': 'Sindhi',
  	'snd': 'Sindhi',
  	'si': 'Sinhala; Sinhalese',
  	'sin': 'Sinhala; Sinhalese',
  	'sk': 'Slovak',
  	'slo/slk*': 'Slovak',
  	'sl': 'Slovenian',
  	'slv': 'Slovenian',
  	'so': 'Somali',
  	'som': 'Somali',
  	'st': 'Sotho, Southern',
  	'sot': 'Sotho, Southern',
  	'es': 'Spanish; Castilian',
  	'spa': 'Spanish; Castilian',
  	'su': 'Sundanese',
  	'sun': 'Sundanese',
  	'sw': 'Swahili',
  	'swa': 'Swahili',
  	'ss': 'Swati',
  	'ssw': 'Swati',
  	'sv': 'Swedish',
  	'swe': 'Swedish',
  	'tl': 'Tagalog',
  	'tgl': 'Tagalog',
  	'ty': 'Tahitian',
  	'tah': 'Tahitian',
  	'tg': 'Tajik',
  	'tgk': 'Tajik',
  	'ta': 'Tamil',
  	'tam': 'Tamil',
  	'tt': 'Tatar',
  	'tat': 'Tatar',
  	'te': 'Telugu',
  	'tel': 'Telugu',
  	'th': 'Thai',
  	'tha': 'Thai',
  	'bo': 'Tibetan',
  	'tib/bod*': 'Tibetan',
  	'ti': 'Tigrinya',
  	'tir': 'Tigrinya',
  	'to': 'Tonga (Tonga Islands)',
  	'ton': 'Tonga (Tonga Islands)',
  	'ts': 'Tsonga',
  	'tso': 'Tsonga',
  	'tn': 'Tswana',
  	'tsn': 'Tswana',
  	'tr': 'Turkish',
  	'tur': 'Turkish',
  	'tk': 'Turkmen',
  	'tuk': 'Turkmen',
  	'tw': 'Twi',
  	'twi': 'Twi',
  	'ug': 'Uighur',
  	'uig': 'Uighur',
  	'uk': 'Ukrainian',
  	'ukr': 'Ukrainian',
  	'ur': 'Urdu',
  	'urd': 'Urdu',
  	'uz': 'Uzbek',
  	'uzb': 'Uzbek',
  	'vi': 'Vietnamese',
  	'vie': 'Vietnamese',
  	'vo': 'Volapuk',
  	'vol': 'Volapuk',
  	'wa': 'Walloon',
  	'wln': 'Walloon',
  	'cy': 'Welsh',
  	'wel/cym*': 'Welsh',
  	'fy': 'Western Frisian',
  	'fry': 'Western Frisian',
  	'wo': 'Wolof',
  	'wol': 'Wolof',
  	'xh': 'Xhosa',
  	'xho': 'Xhosa',
  	'yi': 'Yiddish',
  	'yid': 'Yiddish',
  	'yo': 'Yoruba',
  	'yor': 'Yoruba',
  	'za': 'Zhuang; Chuang',
  	'zha': 'Zhuang; Chuang',
  	'zu': 'Zulu',
  	'zul': 'Zulu'
  }
    #detect languages in all messages
    languages = []
    for data in clean_values:
        if len(data) > 50:
            language_code = detect(data)    
            lang_result = [language_code, data]
            languages.append(language_code)
            #print(lang_result)
    #count frequencies of used languages    
    from collections import Counter
    counts = Counter(languages)
    #print(counts)

    mappd = Counter(" ".join(ele for ele in languages).split())
    
    # getting share of each sentence
    res = {key: val / sum(mappd.values()) for key, val in mappd.items()}
    return res

def get_written_language_diversity(input_data):
    import calendar
    data = read_data(input_data)
    print("data read fin")
    #available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
	#dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]

    available_months = sorted(list(data.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [data[data.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]

    #calculation by month
    monthly_detect_language = dict()
    for i, df in zip(available_months, dfs):
        #print(f"Month # {i}:\n")
        try:
            text =df["Body"].to_list()
            clean_values = clean(text)
            monthly_detect_language.update({str(i) :detect_languages(clean_values)})
            #monthly_detect_language.update({calendar.month_name[i] :detect_languages(clean_values)})

        except KeyError:
            print(f"key error for month {i}")
            continue
    return monthly_detect_language

###normalization 

def balance_written_diversity(seq, k=None):
    from collections import Counter
    from numpy import log
    seq = dict.fromkeys(seq, 1)
    n = sum(seq.values())
    classes = [(clas,float(count)) for clas,count in Counter(seq).items()]
    if not k:
        k = len(classes)
#   H = -sum([(count/n) * log((count/n)) for clas,count in classes]) #shannon entropy
    H = -sum([(count/n) * log((count/n)) for clas,count in classes if count!=0])
    return H/log(k)

def get_n_written_diversity(written_language_diversity, k=None):
    n_written_diversity = dict()
    for month in written_language_diversity.keys() : 
        month_diversity = written_language_diversity[month]
        n_written_diversity.update({month : balance_written_diversity(month_diversity,k)})
    return(n_written_diversity)

############################################
## Language sophistication score 
#######################################
def percentage(part, whole):
  percentage = 100 * float(part)/float(whole)
  return round(percentage)

def get_F_sophistication(text):
	import nltk
	nltk.download('averaged_perceptron_tagger', quiet = True)
	nltk.download('punkt', quiet = True)
	from collections import Counter

	words = nltk.word_tokenize(str(text))
	tags = nltk.pos_tag(words)
	all_tags = len(tags)
	counts = Counter( tag for word,  tag in tags)
	#print(tags)
    #print(all_tags)
	#print(all_tags)

	nouns = counts["NNP"] + counts["NN"] + counts["NNS"]+ counts["NNPS"]
	noun_freq = percentage(nouns, all_tags)
    #print(noun_freq)


	adjectives = counts["JJ"] + counts["JJR"]+ counts["JJS"]
	adjective_freq = percentage(adjectives, all_tags)
    #print("adjective_freq",adjective_freq)

	verbs = counts["VB"] + counts["VBD"] + counts["VBZ"] + counts["VBG"] + counts["VBP"] + counts["VBN"] + counts["MD"]
	verb_freq = percentage(verbs, all_tags)
    #print("verb_freq",verb_freq)

	pronouns = counts["PRP"] + counts["PRP$"] + counts["WP"]
	pronoun_freq = percentage(pronouns, all_tags)
    #print("pronoun_freq",pronoun_freq)

	articles = counts["DT"] + counts["WDT"]
	article_freq = percentage(articles, all_tags)
    #print("article_freq",article_freq)

	prepositions = counts["IN"]
	preposition_freq = percentage(prepositions, all_tags)
    #print("preposition_freq",preposition_freq)

	adverbs = counts["RB"] + counts["RBR"]+counts["RBS"] + counts["WRB"]
	adverb_freq = percentage(adverbs, all_tags)
    #print("adverb_freq",adverb_freq)

	interjections = counts["UH"]
	interjection_freq = percentage(interjections, all_tags)
    #print("interjection_freq", interjection_freq)

    #F will then vary between 0 and 100%.
    #The more formal/sophisticated the language is, the higher the value of F is expected to be.
	F = ((noun_freq + adjective_freq + preposition_freq + article_freq) - (pronoun_freq + verb_freq + adverb_freq + interjection_freq) + 100)/2
    
	return F

def get_members(data):
    sender = data['From'].unique()
    #remove Error 
    sender = sender[sender !="#ERROR!"]
    return sender


def get_language_sophistication(input_data):
    import calendar
    data = read_data(input_data)
    print("data read fin")
    sender = get_members(data)

	#available_months = sorted(list(data.Date_Converted_dt.dt.month.unique()))
	#dfs = [data[data.Date_Converted_dt.dt.month == month] for month in available_months]

    available_months = sorted(list(data.Date_Converted_dt.dt.strftime('%Y-%m').unique()))
    dfs = [data[data.Date_Converted_dt.dt.strftime('%Y-%m')== month] for month in available_months]

    #calculation by month
    monthly_sophistication = dict()

    for i, df in zip(available_months, dfs):
        #loop by member
        within_month = dict()
        for sdr in sender : 
            dfm = df[df['From']==sdr]
            print(i, sdr, len(dfm))
            if len(dfm)==0 :
                continue
            text =dfm["Body"].to_list()
            clean_values = clean(text)
            within_month.update({sdr : get_F_sophistication(clean_values)})
            #print(sdr,":", clean_values)
        #monthly_sophistication.update({calendar.month_name[i] : within_month})
        monthly_sophistication.update({str(i) : within_month})


    return monthly_sophistication

def balance_language_sophistication(seq, k=None):
    from collections import Counter
    from numpy import log
    
    n = sum(seq.values())
    #print("total number:", n)
    classes = [(clas,float(count)) for clas,count in Counter(seq).items()]
    if not k:
        k = len(classes)
#   H = -sum([(count/n) * log((count/n)) for clas,count in classes]) #shannon entropy
    H = -sum([(count/n) * log((count/n)) for clas,count in classes if count!=0])
    return H/log(k)

def get_n_language_sophistication(language_sophistication, k=None):
    n_language_sophistication = dict()
    for month in language_sophistication.keys() : 
        month_diversity = language_sophistication[month]
        n_language_sophistication.update({month : balance_language_sophistication(month_diversity,k)})
    return(n_language_sophistication)

##########################################
#Visualization 
#############################################
def plt_language_sophistication(n_language_sophistication, outdir=None):
	import plotly.graph_objects as go
	# Add data
	month = list(n_language_sophistication.keys())
	scores = list(n_language_sophistication.values())
	fig = go.Figure()
	# Create and style traces
	fig.add_trace(go.Scatter(x=month, y=scores, name='Meeting_intesity_score',
							line=dict(color='red', width=4)))
	fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
	# Edit the layout
	fig.update_layout(#title='<b>Change of Language Sophistication Sub-indicator</b><br>\
	#<i>Language Sophistication diversity is increasing when you start to use more informal communication style (e.g. emodji, less of punctionation, less of articles etc.) <i><br>',
					xaxis_title='Month',
					yaxis_title='Score')
	fig.update_layout(yaxis_range=[0,1])

	# add a footnote to the bottom left 
	#fig.add_annotation(
		#text = ("<i>Scale (min 0, max 1)<i>")
	# , showarrow=False
	#  , x = 0
	#  , y = -0.15
	#  , xref='paper'
	#  , yref='paper' 
	# , xanchor='left'
	# , yanchor='bottom'

	# , font=dict(size=16, color="grey")
	#  , align="left")
	fig.update_layout(margin=dict(t=150))
	fig.update_layout(title={'font': {'size': 20}})


	layout = go.Layout(
		autosize=False,
		width=2000,
		height=2000,
	)
	fig.write_image(outdir+"language_sophistication.png")
	#fig.show()

def plt_language_sophistication_from_stored_data(stored_data=None, select_column="language_sophistication",outdir=None):
    import re
    col_months = [re.findall(r'\d{4}-\d{2}',x)[0] for x in stored_data.columns.tolist() if bool(re.search(r'\d{4}-\d{2}',x))]
    monthly_data= stored_data.loc[stored_data.Indicator==select_column, col_months]
    monthly_data = monthly_data.dropna(axis=1)
    monthly_dic = {col : value.values[0] for col, value in monthly_data.iteritems()}
    plt_language_sophistication(monthly_dic, outdir = outdir)

def plt_written_language_diversity(n_written_language_diversity,outdir=None):
	import plotly.graph_objects as go
	# Add data
	month = list(n_written_language_diversity.keys())
	scores = list(n_written_language_diversity.values())
	fig = go.Figure()
	# Create and style traces
	fig.add_trace(go.Scatter(x=month, y=scores, name='Meeting_intesity_score',
							line=dict(color='blue',dash='dashdot', width=4)))
    
	fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
	# Edit the layout
	fig.update_layout(#title='<b>Change of Written Language Diversity Sub-indicator</b><br>\
	#<i>Written Language diversity is increasing when you start to use more languages in your communication <i><br>',
					xaxis_title='Month',
					yaxis_title='Score')
	fig.update_layout(yaxis_range=[0,1])

	# add a footnote to the bottom left 
	#fig.add_annotation(
		#text = ("<i>Scale (min 0, max 1)<i>")
	# , showarrow=False
	#  , x = 0
	#  , y = -0.15
	#  , xref='paper'
	#  , yref='paper' 
	# , xanchor='left'
	# , yanchor='bottom'

	# , font=dict(size=16, color="grey")
	#  , align="left")
	fig.update_layout(margin=dict(t=150))
	fig.update_layout(title={'font': {'size': 20}})


	layout = go.Layout(
		autosize=False,
		width=2000,
		height=2000,
	)
	fig.write_image(outdir+"written_language_diversity.png")

def plt_written_language_diversity_from_stored_data(stored_data=None, select_column="written_language_diversity",outdir=None):
    import re
    col_months = [re.findall(r'\d{4}-\d{2}',x)[0] for x in stored_data.columns.tolist() if bool(re.search(r'\d{4}-\d{2}',x))]
    monthly_data= stored_data.loc[stored_data.Indicator==select_column, col_months]
    monthly_data = monthly_data.dropna(axis=1)
    monthly_dic = {col : value.values[0] for col, value in monthly_data.iteritems()}
    plt_written_language_diversity(monthly_dic, outdir = outdir)

def plt_communication_diversity(n_written_language_diversity, n_language_sophistication,outdir=None):
	import plotly.graph_objects as go
	# Add data
	month = list(n_language_sophistication.keys())
	lg_sophistication  = list(n_language_sophistication.values())
	written_lg = list(n_written_language_diversity.values())

	fig = go.Figure()

	fig.add_trace(go.Scatter(x=month, y=lg_sophistication, name='Evolution of the combination of <br> formal and informal writing styles',  #change the name of subindicator here
							line=dict(color='red', dash='dashdot', width=3)))
	fig.add_trace(go.Scatter(x=month, y=written_lg, name = 'Evolution of the quantity of <br> languages used in e-mail exchanges',       #change the name of subindicator here
							line=dict(color='blue', dash='dot', width=3)))
	fig.update_xaxes(
        dtick="M1", # sets minimal interval to day
        tickformat="%b %Y", # the date format you want 
    )
	# Edit the layout
	fig.update_layout(#title='<b>Evolution of Communication Diversity Sub-indicators</b>', #remove "#" to use subtitle and description
					xaxis_title='Month',
					yaxis_title='Score')
	fig.update_layout(yaxis_range=[0,1])

	# add a footnote to the bottom left 
	#fig.add_annotation(
		#text = ("<i>Scale (min 0, max 1)<i>")
	# , showarrow=False
	#  , x = 0
	#  , y = -0.15
	#  , xref='paper'
	#  , yref='paper' 
	# , xanchor='left'
	# , yanchor='bottom'

	# , font=dict(size=16, color="grey")
	#  , align="left")
	fig.update_layout(margin=dict(t=100))
	fig.update_layout(title={'font': {'size': 20}})


	layout = go.Layout(
		autosize=False,
		width=2000,
		height=2000,
	)
	fig.write_image(outdir +"communication_diversity.png")
	#fig.show()

def plt_communication_diversity_from_stored_data(stored_data=None, select_column1="written_language_diversity", select_column2="language_sophistication",outdir=None):
    import re
    col_months = [re.findall(r'\d{4}-\d{2}',x)[0] for x in stored_data.columns.tolist() if bool(re.search(r'\d{4}-\d{2}',x))]
    monthly_data1= stored_data.loc[stored_data.Indicator==select_column1, col_months]
    monthly_data1 = monthly_data1.dropna(axis=1)
    monthly_dic1 = {col : value.values[0] for col, value in monthly_data1.iteritems()}
    monthly_data2= stored_data.loc[stored_data.Indicator==select_column2, col_months]
    monthly_data2 = monthly_data2.dropna(axis=1)
    monthly_dic2 = {col : value.values[0] for col, value in monthly_data2.iteritems()}
    plt_communication_diversity(monthly_dic1,monthly_dic2, outdir = outdir)