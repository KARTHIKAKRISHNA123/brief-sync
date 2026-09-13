from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates #UI
from fastapi.responses import HTMLResponse #UI
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Brief-Sync",
    description="A text summarization application powered by HuggingFace T5 Transformers.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summarizer_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summarizer_model")

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device being used:", device)
model.to(device)

#templating
templates = Jinja2Templates(directory=".")

# Input Schema For Dialogue => String
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) #lines"
    text = re.sub(r"\s+", " ", text) # Spaces
    text = re.sub(r"<.*?>", "", text) # HTML tags
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue) # Clean - Preprocess the input dialogue

    # tokenize the input dialogue
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)



    # generate the summary using the fine-tuned model => token ids 
    model.to(device)
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )


    # token ids => decode => summary text  => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# API Endpoints 
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}









