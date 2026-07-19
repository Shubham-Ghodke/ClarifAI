import uvicorn

if __name__ == "__main__":
    print("[RUNNER] Starting uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
