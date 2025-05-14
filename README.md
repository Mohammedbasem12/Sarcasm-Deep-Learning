# Sarcasm Detection (Arabic + English)

## 1. Project Overview  
A bilingual sarcasm detector using a Bi-LSTM + Attention architecture, trained on Arabic and English headlines.  
- Trains on an Arabic Kaggle dataset and an English Kaggle dataset.  
- Uses MUSE-aligned FastText embeddings for cross-lingual alignment.  
- Outputs per-headline sarcasm predictions (probabilities, with optional threshold tuning).

## 2. Repository Structure  
```
├── data/  
│   ├── arsarcasm-v2.csv         # Arabic Kaggle dataset (rename to .csv after download)  
│   └── Sarcasm_Headlines_Dataset.json  # English Kaggle JSONL  
│  
├── notebooks/  
│   └── sarcasm_detector.ipynb    # Full end-to-end notebook with data utils, vocab building, model definition, training, and evaluation  
│  
├── vocab.pkl                     # serialized stoi/itos  
├── requirements.txt              # pinned library versions  
└── README.md                     # this file  
```
Github : https://github.com/Mohammedbasem12/Sarcasm-Deep-Learning
## 3. Requirements  
Pin your environment in `requirements.txt`:  
```
python==3.9
torch==2.0.1
gensim==4.2.0
numpy==1.24.0
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.7.1
nltk==3.8.1
camel-tools==1.0.0
```

## 4. Dataset Links  
- **Arabic (arsarcasm-v2)**: https://www.kaggle.com/datasets/abraralotaibi00/arsarcasm-v2  
- **English (Sarcasm Headlines)**: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection  

After downloading, place:  
- the Arabic CSV as `data/arsarcasm-v2.csv`  
- the English JSONL as `data/Sarcasm_Headlines_Dataset.json`  

 

## 5. Installation & Setup  
1. Clone or unzip the project root.  
2. Create a virtual environment and install dependencies:  
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac 
   venv\Scripts\activate         # Windows
   pip install -r requirements.txt
   ```
3. Download the datasets and embeddings into `./data/` per above.  

## 6. How to Run  
### Launch Notebook (Full Pipeline)  
```bash
cd notebooks/
jupyter lab sarcasm_detector.ipynb
```  

## 7. Notes & Tips  
- If you hit GPU memory limits, reduce batch size or add a sequence-length truncation cell (e.g. max_seq_len=128).  
- Threshold tuning is optional (default cutoff=0.5).  
- All hyperparameters are set in the notebook’s config cell for easy tweaks.  

---

> **Questions?** Reach out to **Mohammad Masalmeh** at **mohammad.masalmeh@bahcesehir.edu.tr**
