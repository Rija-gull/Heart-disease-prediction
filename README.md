# AIC354 Assignment 2 – Machine Learning Projects

**Hafiza Rija Gul · FA24-BSE-044**
AIC354 Machine Learning Fundamentals, COMSATS University Islamabad, Lahore Campus

| Task | Folder | Streamlit main file |
|---|---|---|
| Task 1: Titanic Survival | `titanic_deployment/` | `titanic_deployment/app.py` |
| Task 2: Heart Disease | repository root | `app.py` |

## Running locally

```bash
# Titanic
pip install -r titanic_deployment/requirements.txt
streamlit run titanic_deployment/app.py

# Heart Disease
pip install -r requirements.txt
streamlit run app.py
```

To deploy on Streamlit Community Cloud, create a new app from this repository and set the
main file path to the one listed in the table above. Each project uses the
`requirements.txt` in its own folder.
