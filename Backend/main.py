from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

import io, base64
from PIL import Image

from functions import classify_img, set_response

class ImageJSON(BaseModel):
    img: str

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''
Para rodar a aplicação, deve-se executar no terminal o comando:
    uvicorn --port 5000 --host 127.0.0.1 main:app --reload
'''

@app.post("/classificar")
def classificar(imagem: ImageJSON):
    try:
        
        # img = Image.open('./DSC_1566_FA_180_224x224.jpg')
        img_b64 = imagem.img
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(img_b64, "utf-8"))))

        # Verifica o modo da imagem para ajustá-lo
        if img.mode == 'I;16':
            img.mode = 'I'
            img = img.point(lambda i:i*(1./256)).convert('L')

        classe, probs = classify_img(img)

        response = set_response(classe, probs)

        return response
    
    except:
        raise HTTPException(status_code=404, detail='Student Not Found')


app.openapi_schema = get_openapi(
    title="Estimação do estado do mar a partir do processamento de imagens",
    version="1.0",
    description="Ferramenta para estimação do estado do mar a partir do processamento de imagens. Recebe uma imagem codificada em base64 e retorna a sua classificação.",
    routes=app.routes,
)