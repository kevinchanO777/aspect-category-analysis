from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schemas import ReviewInput
from model import MultiTaskSentimentModel
import uvicorn

app = FastAPI()

# Initialize the trained BERT model
sentiment_model = MultiTaskSentimentModel()


@app.post("/predict/")
async def predict_review(review_input: ReviewInput):
    sentiment_results = sentiment_model.get_sentiment_results(review_input.review)
    return JSONResponse(
        content={"review": review_input.review, "predictions": sentiment_results}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
