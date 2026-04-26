# Sentinel
**Institutional-Grade Equity Analysis Powered by Multi-Agent AI**

Developed by **Noah Cortes, Ethan Goebel, and Stephen Sulimani**

*MGT8803: Generative AI in Finance — Georgia Institute of Technology* *Instructor: Dr. Sudheer Chava*

---

## Project Overview
Sentinel is an advanced AI-driven platform designed to perform rigorous equity research. By leveraging a multi-agent orchestration layer, Sentinel simulates the workflow of a professional investment firm—moving from raw data extraction to peer-reviewed financial reporting.

> **Disclaimer:** Sentinel is a Minimum Viable Product (MVP) and is intended for educational purposes only. It does **not** provide financial advice. Use at your own risk.

---

## System Architecture
Sentinel utilizes a polyglot microservice architecture to balance AI flexibility with computational performance.

| Component | Technology | Primary Role |
| :--- | :--- | :--- |
| **Orchestration** | Python & LangChain | AI agent logic and LLM reasoning. |
| **Reasoning Engine**| Google Gemini | Core LLM for synthesis and analysis. |
| **Quant Engine** | Go (Golang) | High-performance NPV and math calculations. |
| **Frontend** | React | Interactive web interface for report visualization. |
| **Intelligence** | SearXNG | Privacy-focused meta-search for real-time news. |
| **Deployment** | Docker & Compose | Containerized environment for seamless setup. |

---

## The Agentic Workflow
Sentinel employs a "Critic-Loop" design pattern to ensure high-fidelity output.

1.  **Junior Researcher:** Performs the heavy lifting. Scours **SEC EDGAR** filings, executes **NPV calculations** via the Go backend, and synthesizes news via SearXNG.
2.  **The Critic (Senior Researcher):** Reviews the Junior Researcher's findings. This agent identifies logical fallacies, missing data, or calculation errors, passing the report back for iterative refinement.
3.  **Lead Portfolio Manager:** Aggregates the finalized insights and formats the data into a professional **LaTeX** report, which is compiled into a downloadable PDF.

---

## Roadmap & Stretch Goals
We are evolving Sentinel to bridge the gap between retail accessibility and institutional power:

* **Dual-Tier Ecosystem:**
    * **B2C:** Empowering retail investors with institutional-grade tools at an accessible price point.
    * **B2B:** A compliance-focused version for firms, utilizing **locally-hosted LLMs** to protect proprietary data and client privacy.
* **Persistent Knowledge Base:** Implementing a **Retrieval-Augmented Generation (RAG)** pipeline, allowing users to query the entire corpus of research data long after the initial report is generated.
* **Expanded Data Sources:** Integration of alternative data streams and real-time market pricing.

---

## Getting Started
*(Note: Add your specific setup commands here, for example:)*
1. Clone the repository.
2. Ensure Docker and Docker Compose are installed.
3. Create a `.env` file with your Gemini API keys.
4. Run `docker-compose up --build`.

---

## Attribution & License
This project was created as the final capstone for **MGT8803 (Generative AI in Finance)** at the Georgia Institute of Technology.

Released under the [MIT License](LICENSE).

