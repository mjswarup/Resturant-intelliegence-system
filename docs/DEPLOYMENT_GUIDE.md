\# Deployment Guide



\## Local Development



See Quick Start in `README.md`.



\## Docker Deployment



```bash

docker build -t restaurant-intelligence-system .

docker run -d -p 8000:8000 --name ris-api restaurant-intelligence-system

```



Health check:

```bash

curl http://localhost:8000/health

```



\## Environment Variables



This system currently requires no environment variables or secrets — all

models load from local disk paths defined in `src/config/config.py`. If

deploying with a different model storage location (e.g., cloud object storage),

`MODELS\_DIR` in that file is the single point of configuration to change.



\## Cloud Portability Notes



\- The Docker image is self-contained and portable to any container platform

&#x20; (AWS ECS/Fargate, Google Cloud Run, Azure Container Apps, etc.)

\- No database dependency — all data is file-based (CSV + joblib bundles)

\- Recommended minimum resources: 512MB RAM, 1 vCPU (models are small post-optimization; see Phase 20/25 notes on the trimmed image)

\- For production traffic (not demonstrated in this project), remove `--reload`

&#x20; from any uvicorn invocation and consider a process manager (gunicorn with

&#x20; uvicorn workers) for multi-worker concurrency



\## Production Readiness Checklist (Not Fully Implemented — Documented as Future Work)



\- \[ ] Replace CORS `allow\_origins=\["\*"]` with an explicit allowed-origins list

\- \[ ] Add request rate limiting

\- \[ ] Add structured logging / monitoring hooks

\- \[ ] Add authentication if the API is exposed beyond a trusted network

\- \[ ] Set up CI to run `tests/test\_api.py` on every commit

\- \[ ] Pin exact dependency versions (see `requirements\_full.txt` note)



This project was built and validated as a portfolio/academic system — the

above items represent the gap between "demonstrably correct" (achieved) and

"production-hardened" (not attempted, and explicitly out of scope per the

original spec's Phase 24 framing).

