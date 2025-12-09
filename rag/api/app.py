from fastapi import FastAPI
from rag.retrieval.retrieve import retrieve_similar
from ml.inference.inference_service import predict_from_endpoint

app = FastAPI()

@app.post("/predict")
def predict(features: dict):
    """
    Call SageMaker endpoint to produce risk score.
    """
    prediction = predict_from_endpoint(features, endpoint_name="hazard-risk-model")
    return {"prediction": prediction}

@app.post("/ask")
def ask(question: str):
    """
    Embed question → retrieve chunks → call LLM → return answer.
    """
    print("RAG query received:", question)
    # Placeholder retrieval
    return {"answer": "RAG response placeholder", "sources": []}
