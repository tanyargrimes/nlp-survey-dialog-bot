# !/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
PROG8420 - Programming for Big Data

Group 3 Assignment

Created on Tue Aug 11 12:32:58 2020

@authors: 
    Jaibir Singh
    Tanya Grimes
"""


import json
import nltk
import re
import string
import joblib
import tkinter as tk


nltk.download('stopwords')


# Please ensure the current working directory is set to the location of this project folder


# retrieves chatbot responses
with open("bot_response.json") as f:
    response_dict = json.load(f)
    f.close()

# prepare data cleaning techniques
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

# dictionary to store user response ratings
survey_response_rating = {
    'meal': 0,
    'service': 0,
    'comfort': 0,
    'overall': 0
}


# clean data
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


# import vectorizer and model libraries
# requires clean_text function to be predefined
tfidf_vfit = joblib.load('vectorizerFit.joblib')
lgt_model = joblib.load('logisticRegressionModel.joblib')


# process user response
def process_user_response(response, type):
    
    if (type != 'overall'):
        processed = tfidf_vfit.transform([response]).toarray()
        
        # stores array of probabilities for 1 and 5 classification, based on model
        rlist = lgt_model.predict_proba(processed)[0]
        
        # store the probability of the class5 index
        class5_prob = rlist[1]
        
        # using the probability of the class5 probability, the rating of the question
        # can be generated
        if (class5_prob <= 0.2): survey_response_rating[type] = 1
        elif (class5_prob > 0.2 and class5_prob <= 0.4): survey_response_rating[type] = 2
        elif (class5_prob > 0.4 and class5_prob <= 0.6): survey_response_rating[type] = 3
        elif (class5_prob > 0.6 and class5_prob <= 0.8): survey_response_rating[type] = 4
        elif (class5_prob > 0.8): survey_response_rating[type] = 5
        else: survey_response_rating[type] = -1 # accommodates any errors in probabilities returned
             
        bot_response_key = 'star' + str(survey_response_rating[type])
        print(response_dict[bot_response_key],'\n')
        
        if type == 'meal': meal_bresponse.set(response_dict[bot_response_key])
        elif type == 'service': service_bresponse.set(response_dict[bot_response_key])
        elif type == 'comfort': comfort_bresponse.set(response_dict[bot_response_key])
        
        return True
    else:
        overall_rating = (survey_response_rating['meal'] +
                          survey_response_rating['service'] +
                          survey_response_rating['comfort']) / 3
        
        if overall_rating <= 2: survey_response_rating[type] = 1
        elif overall_rating > 2 and overall_rating < 4: survey_response_rating[type] = 2
        elif overall_rating >= 4: survey_response_rating[type] = 3
        else: survey_response_rating[type] = -1
        
        bot_response_key = 'overall' + str(survey_response_rating[type])
        print(response_dict[bot_response_key])
        
        overall_bresponse.set(response_dict[bot_response_key])
        return True
    
    return False



def submit_survey(event = None):
    ''' Displays the current message and sends to the server '''
    
    if meal_value.get() != '' and service_value.get() !=  '' and comfort_value.get() != '':
        msg_error.set('')
        process_user_response(meal_value.get(), 'meal')
        process_user_response(service_value.get(), 'service')
        process_user_response(comfort_value.get(), 'comfort')
        process_user_response('', 'overall')
    else:
        msg_error.set('Please fill in all fields.')
        meal_bresponse.set('')
        service_bresponse.set('')
        comfort_bresponse.set('')
        overall_bresponse.set('')
        
    


#------------------------------------------
# Tkinter GUI Definitions

TK_BTN_BG_COL = '#2d98da'
TK_THEME_COL = '#227093'
TK_BTN_TXT_COL = 'white'
TK_BTN_BG_COL = '#0a3d62'
TK_FRM_BG_COL = '#d1d8e0'
TK_ERR_TXT_COL = '#b33939'
TK_12_STAR = '#b33939'
TK_3_STAR = '#ffb142'
TK_45_STAR = '#218c74'

gui_client = tk.Tk()
gui_client.title('Fiddle Restaurant Survey')


# adds frame to hold the list box and scrollbar
frm_messages = tk.Frame(
    master = gui_client,
    relief = tk.FLAT,
    borderwidth = 20
)
# allows the frame to grow as the widget expands
frm_messages.pack(
    fill = tk.BOTH,
    expand = True
)

# scrolls the frame to see previous messages
srl_scrollbar = tk.Scrollbar(
    master = frm_messages
)


# stores label as a string variable
msg_error = tk.StringVar()

lbl_error = tk.Label(
    master = frm_messages,
    text = '',
    textvariable = msg_error,
    fg = TK_ERR_TXT_COL,
    width = 70,
)
lbl_error.pack()



meal_question = tk.Label(
    master = frm_messages,
    text = '1. How was your meal today?',
    anchor = 'w',
    pady = 10,
    width = 70
)
meal_question.pack()

# stores input as a string variable
meal_value = tk.StringVar()

# adds the input box
meal_input = tk.Entry(
    master = frm_messages,
    textvariable = meal_value,
    width = 80
)

# binds return key to input with submit_sruvey call when pressed
meal_input.bind('<Return>', submit_survey)
meal_input.pack(
    fill = tk.BOTH,
    expand = True
)



# stores input as a string variable
meal_bresponse = tk.StringVar()

meal_response = tk.Label(
    master = frm_messages,
    text = '',
    textvariable = meal_bresponse,
    fg = TK_THEME_COL,
    anchor = 'w',
    width = 70
)
meal_response.pack()



service_question = tk.Label(
    master = frm_messages,
    text = '2. How was the service today?',
    anchor = 'w',
    pady = 10,
    width = 70
)
service_question.pack()

# stores input as a string variable
service_value = tk.StringVar()

# adds the input box
service_input = tk.Entry(
    master = frm_messages,
    textvariable = service_value,
    width = 80
)

# binds return key to input with submit_sruvey call when pressed
service_input.bind('<Return>', submit_survey)
service_input.pack(
    fill = tk.BOTH,
    expand = True
)

# stores input as a string variable
service_bresponse = tk.StringVar()

service_response = tk.Label(
    master = frm_messages,
    text = '',
    textvariable = service_bresponse,
    fg = TK_THEME_COL,
    anchor = 'w',
    #pady = 10,
    width = 70
)
service_response.pack()



comfort_question = tk.Label(
    master = frm_messages,
    text = '3. How was your comfort today?',
    anchor = 'w',
    pady = 10,
    width = 70
)
comfort_question.pack()

# stores input as a string variable
comfort_value = tk.StringVar()

# adds the input box
comfort_input = tk.Entry(
    master = frm_messages,
    textvariable = comfort_value,
    width = 80
)

# binds return key to input with submit_sruvey call when pressed
comfort_input.bind('<Return>', submit_survey)
comfort_input.pack(
    fill = tk.BOTH,
    expand = True
)

# stores input as a string variable
comfort_bresponse = tk.StringVar()

comfort_response = tk.Label(
    master = frm_messages,
    text = '',
    textvariable = comfort_bresponse,
    fg = TK_THEME_COL,
    anchor = 'w',
    width = 70
)
comfort_response.pack()



# define font
overall_font = tk.font.Font(family = 'Helvetica', size = 9, weight = 'bold')

# stores input as a string variable
overall_bresponse = tk.StringVar()

overall_response = tk.Label(
    master = frm_messages,
    text = '',
    textvariable = overall_bresponse,
    fg = TK_BTN_BG_COL,
    anchor = 'w',
    wraplength = 480,
    font = overall_font,
    justify = 'left',
    pady = 20,
    width = 70
)
overall_response.pack()



# adds send button with send_message call on button click
btn_send = tk.Button(
    master = frm_messages,
    text = 'Submit Survey',
    width = 12,
    height = 2,
    bg = TK_BTN_BG_COL,
    fg = TK_BTN_TXT_COL,
    command = submit_survey
)
btn_send.pack()


# opens survey widget and listens for events to prevent another window from opening
gui_client.mainloop()


