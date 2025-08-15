Deploy Link :- https://rag-system-that-combines-traditional-vector-search-fm4ir7abck8.streamlit.app/

# RAG-system-that-combines-traditional-vector-search
🩺 Diabetes Risk Predictor &amp; Meal Planner – A Streamlit app that predicts diabetes risk from health data and uses LangChain + Groq LLM to suggest personalized meals. Supports CSV/Excel recipe uploads, nutritional analysis, ingredient substitutions, and fast recipe retrieval via HuggingFace embeddings.

🩺 Diabetes Risk Predictor & Personalized Meal Planner

This Streamlit-based AI application predicts diabetes risk factors from user health data and generates personalized meal recommendations tailored to their profile.
It integrates LangChain, HuggingFace embeddings, and Groq LLM to retrieve relevant recipes from an uploaded dataset and produce health-conscious meal plans.

🚀 Features

Diabetes Health Data Input – Users can enter health metrics like glucose level, blood pressure, BMI, insulin, and more.

Recipe Dataset Upload – Supports CSV and Excel formats with automatic parsing and error handling.

Vector Store Indexing – Converts recipes into embeddings for fast similarity search.

AI-Powered Meal Recommendations – Uses Groq LLM to suggest 3–5 meals based on the user’s health profile.

Nutritional Analysis – Provides a short nutritional breakdown and dietary guidelines.

Ingredient Substitutions – Suggests alternatives for allergies or restrictions.

Efficient Retrieval – Uses top-matching recipes for relevant and concise AI responses.

🛠 Tech Stack

Frontend: Streamlit

LLM: Groq with deepseek-r1-distill-llama-70b (configurable)

Embeddings: HuggingFace all-MiniLM-L6-v2

Vector Store: LangChain In-Memory Vector Store

Data Processing: Pandas, LangChain Text Splitter

📂 How It Works

Enter Health Profile
Fill in your medical metrics in the form.

Upload Dataset
Provide a recipe dataset (CSV/Excel).

Process Dataset
The system embeds and indexes recipes for fast retrieval.

Generate Plan
Ask for a meal plan—AI returns nutritional info, guidelines, substitutions, and recommendations.
