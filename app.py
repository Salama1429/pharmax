from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

import os
#from typing import Optional, Tuple
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.embeddings import OpenAIEmbeddings

from langchain.prompts import PromptTemplate
import openai
import fitz
from PIL import Image, ImageDraw, ImageFont


os.environ["OPENAI_API_KEY"] = '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'
api_key= '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'
openai.api_key = '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'

# initialize pinecone
import pinecone 
pinecone.init(
    api_key='4beb389a-8d36-4b21-93a7-048d3f11522a',  
    environment='us-west1-gcp-free'  
)

#Vector database for drugbase dataset
from langchain.vectorstores import Pinecone
embeddings = OpenAIEmbeddings()
index_name = "hackthon-sa-pharmaxx"
db = Pinecone.from_existing_index(index_name, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 14})


function_descriptions = [
            {
                "name": "get_idial_prescription",
                "description": "Get the Ideal prescription from given prescription",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Ideal_prescription": {
                            "type": "string",
                            "description": "the idial drugs to treat the patient maybe one or morem, each drug is writen in new line, e.g. Allopurinol 300mg orally once daily",
                        },
                        "Instructions": {
                            "type": "string",
                            "description": "Instructions to the patient."
                        },
                    },
                    "required": ["Ideal_prescription", "Instructions"],
                },
            }
        ]

def get_idial_prescription(diseas):
    
    prompt_template = """you are expert in medicine and pharmacology, Use the following pieces \
of context delimited by delimited by triple backticks i.e. ``` to write the ideal prescription to \
treat the disease at the end, make sure to write the right amount of drugs only.

example = <<[[disease:H. pylori ,idle prescription: 

[Prescription:

Clarithromycin 500mg orally twice daily for 14 days
Metronidazole 500mg orally twice daily for 14 days
Proton pump inhibitor (PPI) (e.g., omeprazole 40mg) orally once daily for 14 days
Continue prescribed antihypertensive medication (Specify medication and dosage)

Instructions:

Take all medications as prescribed by the healthcare professional.
Take clarithromycin, metronidazole, and PPI with food to improve absorption and minimize gastrointestinal side effects.
Complete the full 14-day course of treatment.
Avoid alcohol consumption during the treatment period.
Follow any dietary restrictions or modifications recommended by the healthcare professional.
Maintain good personal hygiene and follow proper food handling practices to reduce the risk of reinfection.]]]>>

```
context:{context}
```


Follow these steps to get the idial prescription for the provided disease
The provided disease will be delimited with four hashtags,\
i.e. #### 

Step 1:#### First decide which drugs is suitable for the provided disease.

Step 2:#### write the idial prescription in the same structure of the provided example.

Step 3:#### decide whether The prescription seems \
excessive and contains multiple medications \
that may not be necessary or recommended in combination or no. 

Step 4:#### if the  prescription \
is not good \
rewrite it and make it idial prescription with the same structure.

step 5:### make sure that the prescriped drugs don't interact with the drugs consumed earlier by the patient, \
and make sure that the prescriped drugs are safe with patient allergies.

Step 6:####: return the idial prescription in the same structure of provided prescription without more words. 

provided disease= ####{question}####

Use the following format:
Step 1: #### <step 1 reasoning>
Step 2: #### <step 2 reasoning>
Step 3: #### <step 3 reasoning>
Step 4: #### <step 4 reasoning>
Step 5: #### <step 5 reasoning>
Step 6: #### <step 6 reasoning>
final idial prescription in the same structure:#### <final idial prescription>

Make sure to include #### to separate every step.

"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm_16k=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    # Chain 1
    chain_type_kwargs = {"prompt": PROMPT}
    chain_one = RetrievalQA.from_chain_type(llm=llm_16k, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

    resultt = chain_one.run(diseas)
    
    user_query = resultt

    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": user_query}],
            functions=function_descriptions,
            function_call="auto",
        )
    ai_response_message = response["choices"][0]["message"]
    Ideal_prescription = eval(ai_response_message['function_call']['arguments']).get("Ideal_prescription")
    Instructions = eval(ai_response_message['function_call']['arguments']).get("Instructions")
    
    final_prescription_= f'''Prescription:\n{Ideal_prescription}\n
Instructions:\n{Instructions}

'''
    
    return final_prescription_



#WORKING GOOD
#MAY BE BETTER TO USE OPENAI NOT CHATOPENAI OR MODIFY THE PROMPT

import gradio as gr
import random
import time

title='''<html>
<head>
  <title>Image and Text Example</title>
  <style>
    .container {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 30vh;
    }

    .image {
      flex: 1;
      text-align: center;
    }

    .text {
      flex: 1;
      text-align: left;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="image">
      <a href="https://ibb.co/wBKzsNY"><img src="https://i.ibb.co/ZGhMm6N/logo.jpg" alt="logo" border="0"></a>
    </div>
    <div class="text">
      <h1>Experience the future of prescription writing with Pharma X.</h1>
    </div>
  </div>
</body>
</html>'''



def transcribe(audio_path):
    audio_file= open(audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

def make_transcription_image(input_text_,input_text_2):
    # Open the image using Pillow
    image = Image.open("Notepad2.png")

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define the text, font, and size
    text = f'''Patient Medical History:\n
    {input_text_2}\n\n
    {input_text_}
    '''
    font = ImageFont.truetype("arial.ttf", size=27)  # Replace with the desired font and size

    # Define the position to insert the text
    position = (50, 700)  # Replace with the desired coordinates

    # Insert the text onto the image
    text_color = (0, 0, 0)
    draw.text(position, text, font=font, fill=text_color,    )

    # Save the modified image
    image_=image.save("output_image.png")
    image_name="output_image.png"
    return image_name


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(title)
 

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            
            patient_history = gr.Textbox(value= '''Chronic disease:Hypertension
Allergies: penicillin
Current medication: Concor 5 mg''',  label= "Patient History")
                             
            
            msg = gr.Textbox(placeholder= "gout",  label= "Type the Diagnosis and press Enter or click Send")

            inputs_audio = gr.Microphone(source="microphone", type="filepath",label="Or say it",
                               interactive=True, streaming=False)
            inputs_audio.change(transcribe, inputs=inputs_audio, outputs=[msg])

            send = gr.Button("Send")
            chatbot = gr.Textbox( label= "Prescription")
            approve = gr.Button("Approve")
        with gr.Column(scale=1, min_width=200):
            prescription = gr.outputs.Image(label="Prescription", type='filepath')
    

    

    send.click(get_idial_prescription, msg, chatbot)
    msg.submit(get_idial_prescription, msg, chatbot)
    approve.click(make_transcription_image ,[chatbot,patient_history], prescription)

    

demo.launch()