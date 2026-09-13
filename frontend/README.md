# Stock ML Lab Front-end

React/Vite interface for the Stock ML Lab API.

## Development

Start the FastAPI backend from the project root:

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Then in another shell:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173/app/
```

Vite proxies `/api`, `/health`, `/docs`, `/redoc`, and `/openapi.json` to
`127.0.0.1:8000`.

## Production build

```bash
npm run build
```

This creates `frontend/dist`. The FastAPI app detects that directory at startup
and serves it at:

```text
http://127.0.0.1:8000/app/
```

The UI reads `/api/v1/capabilities`; Local, Global, and Compare become available
when the backend reports a complete set of global artifacts.


### M13 comparison scope

The `Compare` scope runs Local Ridge vs Global Ridge for regression and Local Logistic vs Global Logistic for classification. It intentionally keeps the model family fixed so that the observed difference isolates training scope.

### M14 evidence layer

The research panel now includes the real M13 OOS scope benchmark and labels its
date-clustered HAC/bootstrap stability analysis as post hoc. M14 is a separate,
locked h=5 confirmation on assets disjoint from the global training universe.

The completed M14 result is shown separately: pooled Ridge was confirmed over
local Ridge for h=5 forecasting loss, while pooled Logistic was not confirmed.
The UI explicitly avoids translating that result into a trading-performance claim.

### M15 release layer

M15 changes packaging and publication readiness, not the scientific result. It
synchronizes the API and UI at version `0.15.0`, keeps the canonical M14 evidence
bundled, and adds automated Python and production-build checks for GitHub.
