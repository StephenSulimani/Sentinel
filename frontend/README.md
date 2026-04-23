# Project Sentinel — frontend

Vite + React + TypeScript + Tailwind, with Shadcn-style primitives (`Button`, `Card`, …) and a terminal / bento dashboard.

## Local development

1. Start APIs (from repo root):

   `docker compose up -d backend ai`

2. Install and run the UI:

   `cd frontend && npm install && npm run dev`

3. Open `http://localhost:5173`.

Optional env (see `.env.example`):

- `VITE_BACKEND_URL` — Go API (default `http://localhost:8081`)
- `VITE_AI_URL` — Flask service (default `http://localhost:8000`)

## Docker (Vite dev server)

From the repo root:

`docker compose up -d frontend`

Then open `http://localhost:5173`. The browser still talks to `localhost:8081` / `8000` on your machine, so keep `backend` and `ai` published on those ports.

## Production build

`npm run build` outputs static files to `dist/`. Serve with any static host; set the same `VITE_*` variables at **build** time so fetches point at your deployed APIs.
