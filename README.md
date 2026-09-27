\# Restaurant Intelligence System



An end-to-end machine learning system providing four core capabilities over

a Zomato-style restaurant dataset (9,551 restaurants, 141 cities):



1\. \*\*Rating Prediction\*\* — regression model estimating a restaurant's likely rating

2\. \*\*Restaurant Recommendation\*\* — content-based filtering by cuisine, location, price, and preferences

3\. \*\*Cuisine Classification\*\* — predicts a restaurant's primary cuisine from its features

4\. \*\*Location Intelligence\*\* — descriptive geospatial analytics by city and locality



Built with a full, documented pipeline: data quality analysis → cleaning →

leakage detection → feature engineering → model comparison → explainability

→ inference engine → FastAPI backend → tested API.



\## Project Structure
restaurant-intelligence-system/

├── api/ # FastAPI backend (routes, schemas, main app)

├── src/ # Core logic: config, data, features, models, inference

├── notebooks/ # Sequential pipeline scripts (Phase 1-20)

├── models/ # Serialized model bundles (versioned, registry-tracked)

├── data/

│ ├── raw/ # Original dataset (not tracked in git)

│ └── processed/ # Cleaned + feature-engineered data

├── reports/ # EDA plots, explainability outputs, phase write-ups

├── tests/ # API test suite

├── Dockerfile

├── requirements.txt

└── README.md


\## Quick Start (Local)



```bash

\# 1. Clone and enter the project

git clone <repo-url>

cd restaurant-intelligence-system



\# 2. Create and activate a virtual environment

python -m venv .venv

.venv\\Scripts\\Activate.ps1      # Windows PowerShell

\# source .venv/bin/activate     # macOS/Linux



\# 3. Install dependencies

pip install -r requirements.txt



\# 4. Place the dataset

\# Copy your CSV to: data/raw/Dataset.csv



\# 5. Run the pipeline (in order) to regenerate models from scratch

python notebooks/03\_data\_cleaning.py

python notebooks/06\_feature\_engineering.py

python notebooks/07\_feature\_selection\_and\_split.py

python notebooks/10\_final\_rating\_models.py

python notebooks/12\_recommendation\_system.py

python notebooks/13\_cuisine\_classification.py

python notebooks/16\_model\_serialization.py



\# 6. Start the API

uvicorn api.main:app --reload



\# 7. Open the interactive docs

\# http://127.0.0.1:8000/docs

```



\## Quick Start (Docker)



```bash

docker build -t restaurant-intelligence-system .

docker run -p 8000:8000 restaurant-intelligence-system

```



Then visit `http://localhost:8000/docs`.



\## API Endpoints



| Method | Path | Purpose |

|---|---|---|

| GET | `/health` | Service health check |

| GET | `/model-info` | Model registry (versions, metrics, status) |

| POST | `/predict/rating` | Predict a restaurant's rating |

| POST | `/recommend` | Get filtered restaurant recommendations |

| POST | `/predict/cuisine` | Predict a restaurant's primary cuisine |

| GET | `/location/analysis` | City/locality-level analytics |



Full interactive documentation (request/response schemas, try-it-out) is

auto-generated at `/docs` when the server is running.



\## Running Tests



```bash

python tests/test\_api.py

```



25 test cases covering valid requests, missing fields, invalid types,

out-of-range values, malformed JSON, and graceful "not found" handling.



\## Model Performance Summary



| Task | Model | Key Metric |

|---|---|---|

| Rating Prediction | Random Forest (realistic, no Votes) | R² = 0.50, MAE = 0.77 |

| Cuisine Classification | XGBoost | Accuracy = 42.9%, F1 = 0.386 |

| Recommendation | Content-based (TF-IDF + filters) | Evaluated qualitatively |

| Location Intelligence | Descriptive analytics | N/A |



See `reports/19\_model\_selection.md` for the full justification, and

`reports/18\_business\_intelligence.md` for derived insights.



\## Known Limitations



\- Heavy geographic bias: \~57% of restaurants are in New Delhi/Gurgaon/Noida

\- Rating model explains \~50% of variance — no review text or time-series data available

\- Cuisine classifier limited by class imbalance (North Indian is \~34% of filtered data)

\- See `reports/19\_model\_selection.md` for full limitations discussion



\## Future Extension Points



The architecture is designed to support additional prediction systems

(demand forecasting, revenue prediction, customer segmentation, etc.)

without restructuring existing code. See `reports/26\_future\_expansion.md`.



\## License



\[Add your license here]

