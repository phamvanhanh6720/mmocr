import re
import base64
from pydantic import BaseModel

import cv2
import numpy as np
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from mmocr.utils.ocr import MMOCR

import warnings
warnings.filterwarnings("ignore")

api = FastAPI(title="Demo", version='0.1.0')
origins = ["*"]
api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

# Load models into memory
ocr = MMOCR(det='TextSnake',
            recog='SATRN')


class ImageInput(BaseModel):
    base64_img: str


@api.post('/recognize-text')
async def recognize_text(img_input: ImageInput):
    img_object = base64.b64decode(img_input.base64_img)
    image = cv2.imdecode(np.fromstring(img_object, np.uint8), cv2.IMREAD_COLOR)
    results = ocr.readtext(image, merge=True)

    return results
