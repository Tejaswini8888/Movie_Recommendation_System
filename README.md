# 🎬 Netflix-Style Movie Recommendation System

A **Netflix-inspired Movie Recommendation System** built using **Python, Streamlit, NLP, and TMDB API**.  
This project demonstrates a **hybrid recommendation engine** with an interactive, modern UI similar to Netflix.

---

## 🌐 Live Demo

🔗 **Try the App Here:**  
https://movierecommendationsystem-project.streamlit.app/

---

## 🚀 Features

- 🎥 Netflix-style dark brown UI
- 🎯 Centered, interactive movie selection dropdown
- 🤖 Hybrid recommendation system  
  - NLP-based content similarity (TF-IDF + Cosine Similarity)  
  - Genre-based similarity
- 🖼️ Movie posters fetched using **TMDB API**
- ✨ Smooth hover animations & interactive UI elements
- ⚡ Optimized performance with Streamlit caching

---

## 🧠 Recommendation Approach

### 1️⃣ Content-Based Filtering (NLP)
- Uses movie **overview text**
- Converts text to numerical vectors using **TF-IDF**
- Measures similarity using **Cosine Similarity**

### 2️⃣ Genre Similarity
- Compares overlap between movie genres
- Helps improve relevance of recommendations

### 🔀 Final Hybrid Score

```
Final Score = 0.7 × NLP Similarity + 0.3 × Genre Similarity
```

Top 5 movies with the highest scores are recommended.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **NLP:** TF-IDF Vectorizer
- **API:** TMDB (The Movie Database)
- **Styling:** Custom CSS (Netflix-style theme)

---

## 📂 Project Structure

```
├── app.py                 # Main Streamlit application
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── .streamlit/
    └── secrets.toml       # TMDB API key (not pushed to GitHub)
```

---

## 🔑 TMDB API Setup

1. Create an account at https://www.themoviedb.org/
2. Generate an API key
3. Create the file:

```
.streamlit/secrets.toml
```

4. Add the following:

```toml
TMDB_API_KEY = "your_api_key_here"
```

⚠️ **Important:**  
Do **not** push `secrets.toml` to GitHub.

---

## ▶️ Run Locally

```bash
# Clone the repository
git clone https://github.com/Tejaswini8888/Movie_Recommendation_System.git

# Navigate into the project
cd Movie_Recommendation_System

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📸 Screenshots

<img width="966" height="748" alt="Home Screen" src="https://github.com/user-attachments/assets/eab32f65-3644-47e7-92c9-db01100e3e47" />

<img width="661" height="590" alt="Movie Selection" src="https://github.com/user-attachments/assets/15815d0c-22ce-44ad-a75d-7d7404c6aba6" />

<img width="1847" height="853" alt="Recommendations" src="https://github.com/user-attachments/assets/f66baaab-272b-4f11-8a12-08356de315b4" />

---

## 👩‍💻 Author

**Tejaswini Madarapu**

- GitHub: https://github.com/Tejaswini8888  
- LinkedIn: https://www.linkedin.com/in/tejaswini-madarapu/

---

## ⭐ Acknowledgements

- TMDB for movie data & posters
- Streamlit for the UI framework
- Scikit-learn for machine learning utilities

---

## 📜 License

This project is licensed under the **MIT License**.

---

✨ **If you like this project, don’t forget to give it a ⭐ on GitHub!**
