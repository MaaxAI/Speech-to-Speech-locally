import sys
sys.path.append('/home/azureuser/Verbi/Kokoro-82M')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from config import Config
import torch
import uuid
import soundfile as sf
from kokoro import generate
from models import build_model

app = FastAPI()

class TextToSpeechRequest(BaseModel):
    """
    Model representing a text-to-speech request.
    
    Attributes:
        text (str): The text to convert to speech.
        language (str): The language of the text.
        accent (str): The accent to use for the speech.
        speed (float): The speed of the speech.
        filename (str): The desired name for the output audio file.
    """
    text: str
    language: str = 'EN'
    accent: str = 'af_sky'  # Default accent for Kokoro-82M
    speed: float = 1.0
    filename: str = Field(default_factory=lambda: f"{uuid.uuid4()}.wav")

def get_device():
    """
    Determine the appropriate device for running the TTS model.
    
    Returns:
        str: The device to use ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# Initialize the TTS model
device = get_device()  # Determine the appropriate device
#model_path = "kokoro-v0_19.pth"
#model = build_model(model_path, device).to(device)
model_path = "/home/azureuser/Verbi/Kokoro-82M/kokoro-v0_19.pth"
model_dict = build_model(model_path, device)
print(model_dict.keys())  # Print the keys to understand the structure

# Assuming 'predictor' is the model
model = model_dict['predictor'].to(device)

# Load the model state dictionary
model_state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(model_state_dict, strict=False)

# Load the voice pack
voice_name = "af_sky"
voicepack_path = f"/home/azureuser/Verbi/Kokoro-82M/voices/{voice_name}.pt"
voicepack = torch.load(voicepack_path, map_location=device)

@app.post("/generate-audio/")
def generate_audio(request: TextToSpeechRequest):
    """
    Generate an audio file from the given text.

    Args:
        request (TextToSpeechRequest): The request containing text and other parameters.

    Returns:
        dict: A dictionary containing a message and the file path of the generated audio.
    
    Raises:
        HTTPException: If there is an error during audio generation.
    """
    try:
        # Use the provided filename or generate a unique one
        output_filename = request.filename
        
        # Generate the audio file
        audio = []
        for chunk in request.text.split("."):
            if len(chunk) < 2:
                continue
            snippet, _ = generate(model, chunk, voicepack, lang=request.language)
            audio.extend(snippet)
        
        sf.write(output_filename, audio, 24000)
        
        return {"message": "Audio file generated successfully", "file_path": output_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.TTS_PORT_LOCAL)