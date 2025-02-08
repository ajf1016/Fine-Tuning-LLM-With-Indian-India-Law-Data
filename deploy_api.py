from fastapi import FastAPI

app = FastAPI()


@app.post("/generate_fir")
def generate_fir(complaint: str):
    response = pipe(complaint, max_length=500)
    return {"fir_text": response[0]["generated_text"]}


# Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
