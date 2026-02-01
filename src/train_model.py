#!/usr/bin/env python3
"""
ML Pipeline - Triage Automatico dei Ticket
Addestra modelli di classificazione per categoria e priorità
"""

import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

class TicketMLPipeline:
    def __init__(self, data_path=None, models_path=None):
        """
        Inizializza la pipeline ML
        
        Args:
            data_path: Percorso al dataset CSV
            models_path: Cartella dove salvare i modelli
        """
        self.data_path = data_path or Path(__file__).parent.parent / "data" / "tickets_dataset.csv"
        self.models_path = models_path or Path(__file__).parent.parent / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Modelli addestrati
        self.category_model = None
        self.priority_model = None
        self.vectorizer = None
        
        # Metriche
        self.category_metrics = {}
        self.priority_metrics = {}
    
    def load_data(self):
        """Carica il dataset"""
        if not self.data_path.exists():
            print(f"❌ Dataset non trovato: {self.data_path}")
            print("Esegui prima: python src/dataset_generator.py")
            return None
        
        print(f"✓ Caricamento dataset: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"  Dimensioni: {df.shape}")
        print(f"  Categorie: {df['category'].value_counts().to_dict()}")
        print(f"  Priorità: {df['priority'].value_counts().to_dict()}")
        
        return df
    
    def preprocess_text(self, text):
        """
        Preprocessing del testo:
        - lowercase
        - rimozione punteggiatura
        - rimozione caratteri speciali
        """
        if pd.isna(text):
            return ""
        
        # Converte in lowercase
        text = text.lower()
        
        # Rimuove punteggiatura e caratteri speciali
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Rimuove spazi multipli
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_features(self, df):
        """Prepara il testo combinato per la vettorizzazione"""
        print("✓ Preprocessing testo...")
        
        # Combina titolo e corpo
        df['full_text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
        
        # Applica preprocessing
        df['clean_text'] = df['full_text'].apply(self.preprocess_text)
        
        print(f"  Testo pulito creato per {len(df)} ticket")
        
        return df['clean_text'], df['category'], df['priority']
    
    def train_vectorizer(self, texts):
        """Addestra il TfidfVectorizer"""
        print("✓ Addestramento TfidfVectorizer...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limita features per evitare overfitting
            stop_words=None,    # Non usiamo stop words italiane per ora
            ngram_range=(1, 2), # Unigrammi e bigrammi
            min_df=2,           # Ignora termini con frequenza < 2
            max_df=0.8          # Ignora termini con frequenza > 80%
        )
        
        X = self.vectorizer.fit_transform(texts)
        print(f"  Features estratte: {X.shape[1]}")
        print(f"  Documenti: {X.shape[0]}")
        
        return X
    
    def train_category_model(self, X_train, y_train, X_test, y_test):
        """Addestra il modello per la classificazione della categoria"""
        print("✓ Addestramento modello Categoria...")
        
        # Prova Logistic Regression (più veloce e interpretabile)
        try:
            self.category_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='ovr'
            )
        except TypeError:
            # Per versioni più vecchie di scikit-learn
            self.category_model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        
        self.category_model.fit(X_train, y_train)
        
        # Valutazione
        y_pred = self.category_model.predict(X_test)
        
        self.category_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"  Accuracy: {self.category_metrics['accuracy']:.3f}")
        print("  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return y_pred
    
    def train_priority_model(self, X_train, y_train, X_test, y_test):
        """Addestra il modello per la classificazione della priorità"""
        print("✓ Addestramento modello Priorità...")
        
        # Prova Random Forest (buona per dati categorici)
        self.priority_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        self.priority_model.fit(X_train, y_train)
        
        # Valutazione
        y_pred = self.priority_model.predict(X_test)
        
        self.priority_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"  Accuracy: {self.priority_metrics['accuracy']:.3f}")
        print("  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return y_pred
    
    def save_models(self):
        """Salva modelli e vectorizer"""
        print("✓ Salvataggio modelli...")
        
        # Salva vectorizer
        vectorizer_path = self.models_path / "tfidf_vectorizer.joblib"
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"  Vectorizer salvato: {vectorizer_path}")
        
        # Salva modello categoria
        category_model_path = self.models_path / "category_classifier.joblib"
        joblib.dump(self.category_model, category_model_path)
        print(f"  Modello categoria salvato: {category_model_path}")
        
        # Salva modello priorità
        priority_model_path = self.models_path / "priority_classifier.joblib"
        joblib.dump(self.priority_model, priority_model_path)
        print(f"  Modello priorità salvato: {priority_model_path}")
        
        # Salva metriche
        metrics = {
            'category_metrics': self.category_metrics,
            'priority_metrics': self.priority_metrics
        }
        metrics_path = self.models_path / "model_metrics.joblib"
        joblib.dump(metrics, metrics_path)
        print(f"  Metriche salvate: {metrics_path}")
    
    def train_pipeline(self):
        """Esegue l'intera pipeline di training"""
        print("=" * 60)
        print("🚀 INIZIO TRAINING ML PIPELINE")
        print("=" * 60)
        
        # 1. Carica dati
        df = self.load_data()
        if df is None:
            return False
        
        # 2. Prepara features
        X_text, y_category, y_priority = self.prepare_features(df)
        
        # 3. Split train/test
        print("✓ Split train/test (80/20)...")
        X_train_text, X_test_text, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
            X_text, y_category, y_priority, 
            test_size=0.2, 
            random_state=42,
            stratify=y_category  # Stratifica per categoria
        )
        
        print(f"  Train: {len(X_train_text)}")
        print(f"  Test: {len(X_test_text)}")
        
        # 4. Vettorizzazione
        X_train = self.train_vectorizer(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        
        # 5. Addestra modelli
        print("\n" + "=" * 40)
        print("📊 ADDESTRAMENTO MODELLI")
        print("=" * 40)
        
        # Categoria
        cat_pred = self.train_category_model(X_train, y_cat_train, X_test, y_cat_test)
        
        print("\n" + "-" * 30)
        
        # Priorità
        prio_pred = self.train_priority_model(X_train, y_prio_train, X_test, y_prio_test)
        
        # 6. Salva modelli
        print("\n" + "=" * 40)
        print("💾 SALVATAGGIO MODELLI")
        print("=" * 40)
        self.save_models()
        
        # 7. Riassunto finale
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETATO")
        print("=" * 60)
        print(f"Modello Categoria - Accuracy: {self.category_metrics['accuracy']:.3f}")
        print(f"Modello Priorità - Accuracy: {self.priority_metrics['accuracy']:.3f}")
        print(f"Modelli salvati in: {self.models_path}")
        
        return True

def main():
    """Funzione principale"""
    pipeline = TicketMLPipeline()
    success = pipeline.train_pipeline()
    
    if success:
        print("\n🎉 Pipeline ML completata con successo!")
        print("Prossimi passi:")
        print("1. python src/dashboard.py - per la dashboard")
        print("2. python src/predict.py - per fare predizioni su nuovi ticket")
    else:
        print("\n❌ Pipeline ML fallita")
        print("Assicurati di aver generato il dataset prima di procedere.")

if __name__ == "__main__":
    main()