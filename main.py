from fastapi import FastAPI
from InstructorEmbedding import INSTRUCTOR
import logging
from schemas.EmbassPayload import EmbassPayload
from utils.pair_sentences_with_instruction import (
    pair_sentences_with_instruction
)

app = FastAPI()

logging.basicConfig(level=logging.INFO)


@app.on_event("startup")
async def startup_event():
    logging.info("Loading Instructor Embedding Model")
    global model
    model = INSTRUCTOR('hkunlp/instructor-large')
    logging.info("Instructor Embedding Model Loaded")


@app.get("/")
def root():
    return {"Hello": "World"}


@app.post("/embed")
async def embed(payload: EmbassPayload):
    logging.info("EMBED: Received payload")
    if payload.texts is None:
        logging.error("EMBED: texts is required")
        return {"error": "texts is required"}
    if payload.instruction is None:
        logging.warning("EMBED: instruction is not provided, using default")
        payload.instruction = "Represent the query for retrieval"

    logging.info("EMBED: embedding texts")
    embddings = model.encode(pair_sentences_with_instruction(
        payload.texts, payload.instruction))
    logging.info("EMBED: embedding texts done, returning to client")
    return {"embeddings": embddings.tolist()}
