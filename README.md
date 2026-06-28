# Telco Customer Churn: End-to-End MLOps & Business Impact Project

Welcome to my portfolio project! I built this project to systematically predict customer churn in a telecommunications company and solve this problem through an end-to-end machine learning architecture. 

During development, I loved putting on two different hats: **The Data Scientist** and **The Machine Learning Engineer**. My goal was not just to achieve high model metrics, but to deliver a concrete, data-driven blueprint for business stakeholders to optimize retention strategies, all while establishing a microservice architecture ready for production integration.

---

## 1. The Data Scientist Hat (Research & Business Value)

To maintain strict methodological honesty and prevent data leakage, I carefully locked all preprocessing pipelines within the training cross-validation loops. 

### Experimentation & Financial Simulation ($217k Net Profit)
I evaluated multiple classifiers and utilized PyCaret's optimization frameworks. Instead of stopping at standard metrics like F1-Score, I translated the model's performance into a real-world **Financial Simulation**. 
By calculating the promotion costs (for false alarms) versus the revenue saved from correctly persuaded customers, **this model is projected to return a net profit of ~$217,000 USD to the company.**

### Probability Threshold Optimization
Rather than blindly accepting the default 0.50 decision boundary, I optimized the threshold to locate the exact sweet spot for budget efficiency. This successfully reduced false alarms while keeping recall intact, maximizing the Return on Investment (ROI) of the retention campaigns.

### Model Explainability & Business Insights (SHAP)
I strongly believe a model shouldn't be a "black box." Using SHAP values, I extracted actionable strategic insights from the data:
* **Structural Lock-in:** Contract type and Tenure Months emerged as the absolute most critical drivers of retention.
* **The Price Paradox:** Customers do not leave purely because of price; they churn primarily when they lack a structural or contractual commitment.
* **The Fiber-Optic Red Flag:** Internet Service via Fiber Optic significantly *increases* churn risk. This serves as a massive operational red flag, pointing to potential localized service quality issues.

---

## 2. The Machine Learning Engineer Hat (Deployment & MLOps)

Instead of building complex, high-maintenance pipelines, I deployed a "modular" architecture with a clear separation of responsibilities:

* **Microservice Architecture:** I used **FastAPI** for the high-performance backend prediction REST API, and **Streamlit** for the interactive frontend.
* **Containerization:** I packaged the complete pipeline into an isolated **Docker** container (managed via `docker-compose`), ensuring consistent, error-free execution across all cloud platforms.
* **Large File Management:** To prevent repository bloat, I tracked and managed large model (`.pkl`) files using **Git LFS**.
* **Continuous Integration (CI/CD):** I integrated automated unit tests via **GitHub Actions** to ensure system integrity with every new code commit.
* **Model Registry (Versioning):** I utilized the GitHub Releases feature (v1.0.0) as a practical Model Registry to host and track the finalized, production-ready model assets.

---

## Setup & Execution (Local Docker Environment)

I designed the application to be easily deployed on any local machine or server using Docker. Feel free to clone and test it yourself!

### Prerequisites
* **Docker** and **Docker Compose** must be installed.
* **Git LFS** is required when cloning the repository (`git lfs install`).

### Deployment Steps

1. **Clone the Repository:**
   
```bash
   git clone [https://github.com/Bahri81/Telco_Churn_MLOps_Project.git](https://github.com/Bahri81/Telco_Churn_MLOps_Project.git)
   cd Telco_Churn_MLOps_Project

```bash
   docker-compose up --build

Access the Application:

Interactive Streamlit Frontend: http://localhost:8501

FastAPI Backend Documentation (Swagger UI): http://localhost:8000/docs

Conclusion
Beyond just predicting customer profiles likely to leave, I wanted this project to provide a fully-fledged product that translates predictive analytics into actionable business strategies. It bridges the gap between deep analytical research (SHAP, $217k Budget Simulation) and modern software engineering principles (Git LFS, CI/CD, Containerization).

Thank you for checking it out!
