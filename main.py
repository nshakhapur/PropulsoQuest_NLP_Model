import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased')

# Define a function for question-answering
def answer_question(user_input):
    data = [
    ("What is the primary focus of the paper titled 'Ignition-Cycle Investigation of a LaB6 Hollow Cathode for 3–5-Kilowatt Hall Thruster'?",
     "The primary focus of the paper is the investigation of a newly designed heaterless LaB6 hollow cathode over 10,000 ignition cycles."),

    ("Could you provide more details about what was monitored during the 10,000 ignition cycles of the LaB6 hollow cathode?",
     "During the 10,000 ignition cycles, the paper monitored the current-voltage characteristics and upstream pressure of the hollow cathode."),

    ("What heating approach was used to maintain stable performance of the LaB6 hollow cathode, and what was the observed variation in anode voltage during testing?",
     "The paper mentions using a high-voltage cold ignition approach and an arc discharge mode for heating, and it notes that the anode voltage varied within 4 V during the test."),

    ("The text mentions chamfering of certain components. What components were chamfered, and what was the purpose of this chamfering?",
     "The downstream end of the emitter and the orificed cathode plate were slightly chamfered. This was done to reduce ignition erosion."),

    ("Could you summarize the key reasons provided in the paper for the reduction of ignition erosion and the stable performance of the LaB6 hollow cathode?",
     "The paper analyzes the dominant reasons for these outcomes but does not specify them in the given text.")

    
 ]

    additional_data = [
    ("In the study regarding the stability of solid rocket motors, what is the name of the experimental setup used to measure the contribution of aluminum combustion to stability?",
     "The experimental setup is referred to as the 'velocity-coupled T-burner.'"),

    ("What does the study on solid rocket motors reveal about the impact of aluminum combustion on stability, particularly in aluminized propellants?",
     "The study demonstrates that aluminized propellants have a strong destabilizing contribution, while nonaluminized propellants do not. This destabilizing behavior is influenced by the size distribution of aluminum agglomerates."),

    ("What is the main challenge addressed in the study related to paraffin-based solid fuels, and how is it overcome?",
     "The main challenge is the high temperature of the binder degradation in paraffin-based solid fuels. This challenge is overcome by using a metallic complex as a catalyst, which selectively lowers the temperature of polymer degradation, improving the compatibility between binder and paraffin."),

    ("What were the effects of the copper catalyst on the regression rate of the fuel grain in the study of paraffin-based solid fuels?",
     "The inclusion of the copper catalyst substantially improved the regression rate of the fuel grain, suggesting it as a promising alternative for hybrid rocket motors."),

    ("In the examination of burning rates for nanoaluminum-water solid propellants, what were the pressure exponents observed for propellants containing ALEX, L-ALEX, and bimodal compositions?",
     "The pressure exponents observed were 𝑛=0.34 for ALEX, 𝑛=0.5 for L-ALEX, and 𝑛=0.28 for bimodal compositions, indicating how the burning rates change with pressure for these propellants.")
 ]


    data += additional_data

    # Extract questions and answers
    questions = [item[0] for item in data]
    answers = [item[1] for item in data]

    # Encode user input
    input_tokens = tokenizer.tokenize(user_input)
    max_length = 512
    input_tokens = input_tokens + [tokenizer.pad_token] * (max_length - len(input_tokens))
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [1 if token != tokenizer.pad_token else 0 for token in input_tokens]
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)

    # Make predictions
    output = model(input_ids, attention_mask=attention_mask)
    prediction = output[0].argmax(dim=1).item()

    return answers, prediction

# Create a Streamlit app
st.title("PropulsoQuest")
st.subheader("An Innovative Natural Language Processor Model for Propulsion Technologies")
st.subheader("This model is built on Scientific Research Papers and Studies on Propulsion Technologies.")

st.markdown("#### Research Paper Summary:")

txt = """###
Researchers are delving into the behavior of advanced materials in propulsion systems. For instance, they are scrutinizing the operational characteristics of LaB6 hollow cathodes in Hall thrusters, focusing on ignition cycles. Additionally, they are examining the destabilizing effects of aluminum combustion in solid rocket motors, particularly concerning aluminum agglomerate size distribution.

Furthermore, there is a notable effort in the development of innovative solid fuels for rocket propulsion, with an emphasis on paraffin-based formulations. Special attention is given to the addition of metallic complexes as catalysts to mitigate the high-temperature degradation of polymer binders. This enhances the compatibility of binders with paraffin, even in extreme thermal conditions.

Moreover, the burning rates of nanoaluminum particles mixed with water-based rocket propellants are investigated across varying pressures. This exploration aids in comprehending the combustion efficiency and performance of these propellants under diverse environmental conditions. These multidisciplinary studies aim to advance the efficiency, stability, and safety of propulsion systems, contributing to the forefront of aerospace technology.
"""
st.markdown(txt)

# User input for questions

user_input = st.text_input("Please   enter    a    query    regarding    propulsion:")

# Process user input and get an answer
if user_input:
    answers, prediction = answer_question(user_input)
    st.markdown("## Answer is: ")
    st.markdown("### {}".format(answers[prediction - 1]))
