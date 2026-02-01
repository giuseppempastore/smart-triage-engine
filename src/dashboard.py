#!/usr/bin/env python3
"""
Dashboard Interattiva - Triage Automatico dei Ticket
Interfaccia grafica per la classificazione automatica di ticket
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione Streamlit
st.set_page_config(
    page_title="Smart Triage Engine",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SmartTriageDashboard:
    def __init__(self):
        """Inizializza la dashboard"""
        self.models_path = Path(__file__).parent.parent / "models"
        self.category_model = None
        self.priority_model = None
        self.vectorizer = None
        
        # Carica i modelli
        self.load_models()
    
    def load_models(self):
        """Carica i modelli salvati"""
        try:
            # Carica vectorizer
            vectorizer_path = self.models_path / "tfidf_vectorizer.joblib"
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Carica modello categoria
            category_model_path = self.models_path / "category_classifier.joblib"
            self.category_model = joblib.load(category_model_path)
            
            # Carica modello priorità
            priority_model_path = self.models_path / "priority_classifier.joblib"
            self.priority_model = joblib.load(priority_model_path)
            
            st.success("✅ Modelli caricati correttamente!")
            
        except FileNotFoundError as e:
            st.error(f"❌ Modelli non trovati: {e}")
            st.info("Esegui prima: python src/train_model.py")
            st.stop()
        except Exception as e:
            st.error(f"❌ Errore nel caricamento modelli: {e}")
            st.stop()
    
    def preprocess_text(self, text):
        """
        Preprocessing del testo (stesso della pipeline di training)
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
    
    def get_feature_importance(self, text, model, vectorizer, top_n=5):
        """
        Calcola le parole più influenti per la predizione
        
        Args:
            text: Testo da analizzare
            model: Modello addestrato
            vectorizer: TfidfVectorizer
            top_n: Numero di parole da restituire
        
        Returns:
            Lista di tuple (parola, importanza)
        """
        # Vettorizza il testo
        X = vectorizer.transform([text])
        
        # Calcola importanza feature (coefficienti del modello)
        if hasattr(model, 'coef_'):
            # Per Logistic Regression
            feature_importance = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        elif hasattr(model, 'feature_importances_'):
            # Per Random Forest
            feature_importance = model.feature_importances_
        else:
            return []
        
        # Ottieni nomi feature
        feature_names = vectorizer.get_feature_names_out()
        
        # Associa importanza a nomi feature
        feature_importance_pairs = list(zip(feature_names, feature_importance))
        
        # Ordina per importanza (valore assoluto)
        feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Restituisce top N
        return feature_importance_pairs[:top_n]
    
    def predict_single_ticket(self, title, body):
        """
        Predice categoria e priorità per un singolo ticket
        
        Args:
            title: Titolo del ticket
            body: Corpo del ticket
        
        Returns:
            Dict con predizioni e feature importance
        """
        # Combina e preprocessa testo
        full_text = f"{title} {body}"
        clean_text = self.preprocess_text(full_text)
        
        if not clean_text.strip():
            return None
        
        # Vettorizza
        X = self.vectorizer.transform([clean_text])
        
        # Predizioni
        category_pred = self.category_model.predict(X)[0]
        priority_pred = self.priority_model.predict(X)[0]
        
        # Probabilità
        category_proba = self.category_model.predict_proba(X)[0]
        priority_proba = self.priority_model.predict_proba(X)[0]
        
        # Feature importance
        category_features = self.get_feature_importance(
            clean_text, self.category_model, self.vectorizer, top_n=5
        )
        priority_features = self.get_feature_importance(
            clean_text, self.priority_model, self.vectorizer, top_n=5
        )
        
        return {
            'category': category_pred,
            'priority': priority_pred,
            'category_proba': category_proba,
            'priority_proba': priority_proba,
            'category_features': category_features,
            'priority_features': priority_features,
            'input_text': clean_text
        }
    
    def predict_batch(self, df):
        """
        Predice categoria e priorità per un batch di ticket
        
        Args:
            df: DataFrame con colonne 'title' e 'body'
        
        Returns:
            DataFrame con predizioni aggiunte
        """
        results = []
        
        for idx, row in df.iterrows():
            title = row.get('title', '')
            body = row.get('body', '')
            
            prediction = self.predict_single_ticket(title, body)
            
            if prediction:
                results.append({
                    'id': row.get('id', idx + 1),
                    'title': title,
                    'body': body,
                    'predicted_category': prediction['category'],
                    'predicted_priority': prediction['priority'],
                    'category_confidence': max(prediction['category_proba']),
                    'priority_confidence': max(prediction['priority_proba'])
                })
            else:
                results.append({
                    'id': row.get('id', idx + 1),
                    'title': title,
                    'body': body,
                    'predicted_category': 'N/A',
                    'predicted_priority': 'N/A',
                    'category_confidence': 0.0,
                    'priority_confidence': 0.0
                })
        
        return pd.DataFrame(results)

def main():
    """Funzione principale della dashboard"""
    
    # Header
    st.title("🎫 Smart Triage Engine")
    st.markdown("**Classificazione automatica di ticket per Categoria e Priorità**")
    st.markdown("---")
    
    # Inizializza dashboard
    dashboard = SmartTriageDashboard()
    
    # Sidebar per navigazione
    st.sidebar.title("Navigazione")
    page = st.sidebar.selectbox(
        "Seleziona funzionalità:",
        ["🔍 Singolo Ticket", "📊 Batch Prediction", "📈 Metriche Modelli"]
    )
    
    if page == "🔍 Singolo Ticket":
        # Sezione Singolo Ticket
        st.header("🔍 Classificazione Singolo Ticket")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input(
                "Titolo del Ticket:",
                placeholder="Es: Problema di connessione al server"
            )
        
        with col2:
            priority_hint = st.selectbox(
                "Priorità (solo per riferimento):",
                ["Bassa", "Media", "Alta"],
                index=1
            )
        
        body = st.text_area(
            "Descrizione del Ticket:",
            height=200,
            placeholder="Es: Non riesco ad accedere al server da questa mattina..."
        )
        
        if st.button("🚀 Analizza Ticket", type="primary"):
            if not title.strip() and not body.strip():
                st.warning("⚠️ Inserisci almeno il titolo o la descrizione del ticket.")
            else:
                with st.spinner("Analisi in corso..."):
                    prediction = dashboard.predict_single_ticket(title, body)
                
                if prediction:
                    # Mostra risultati
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"**Categoria Predetta:** {prediction['category']}")
                        st.info(f"**Priorità Suggerita:** {prediction['priority']}")
                    
                    with col2:
                        # Visualizza confidenza
                        cat_conf = max(prediction['category_proba'])
                        prio_conf = max(prediction['priority_proba'])
                        
                        st.metric("Confidenza Categoria", f"{cat_conf:.1%}")
                        st.metric("Confidenza Priorità", f"{prio_conf:.1%}")
                    
                    st.markdown("---")
                    
                    # Feature Importance
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔤 Parole Chiave - Categoria")
                        if prediction['category_features']:
                            for word, importance in prediction['category_features']:
                                st.write(f"- **{word}**: {abs(importance):.3f}")
                        else:
                            st.write("Nessuna parola significativa trovata.")
                    
                    with col2:
                        st.subheader("🔤 Parole Chiave - Priorità")
                        if prediction['priority_features']:
                            for word, importance in prediction['priority_features']:
                                st.write(f"- **{word}**: {abs(importance):.3f}")
                        else:
                            st.write("Nessuna parola significativa trovata.")
                    
                    # Testo preprocessato
                    with st.expander("📄 Testo Preprocessato"):
                        st.code(prediction['input_text'], language="text")
                
                else:
                    st.error("❌ Impossibile analizzare il ticket. Verifica il testo inserito.")
    
    elif page == "📊 Batch Prediction":
        # Sezione Batch Prediction
        st.header("📊 Classificazione Batch di Ticket")
        
        st.info("""
        **Istruzioni:**
        1. Carica un file CSV con colonne 'title' e 'body'
        2. La dashboard analizzerà ogni ticket
        3. Potrai scaricare i risultati con le predizioni
        """)
        
        uploaded_file = st.file_uploader(
            "Carica file CSV con ticket:",
            type=['csv'],
            help="Il file deve contenere le colonne 'title' e 'body'"
        )
        
        if uploaded_file is not None:
            try:
                # Leggi file
                df = pd.read_csv(uploaded_file)
                st.success(f"✓ File caricato: {len(df)} ticket")
                
                # Mostra anteprima
                st.subheader("Anteprima Dati")
                st.dataframe(df.head())
                
                # Verifica colonne
                if 'title' not in df.columns or 'body' not in df.columns:
                    st.error("❌ Il file deve contenere le colonne 'title' e 'body'")
                else:
                    if st.button("🚀 Avvia Classificazione Batch", type="primary"):
                        with st.spinner("Classificazione in corso..."):
                            results_df = dashboard.predict_batch(df)
                        
                        # Mostra risultati
                        st.subheader("Risultati della Classificazione")
                        st.dataframe(results_df)
                        
                        # Statistiche
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Ticket Analizzati", len(results_df))
                            st.metric("Ticket con Predizione", len(results_df[results_df['predicted_category'] != 'N/A']))
                        
                        with col2:
                            cat_counts = results_df['predicted_category'].value_counts()
                            prio_counts = results_df['predicted_priority'].value_counts()
                            
                            st.write("**Distribuzione Categorie:**")
                            for cat, count in cat_counts.items():
                                st.write(f"- {cat}: {count}")
                            
                            st.write("**Distribuzione Priorità:**")
                            for prio, count in prio_counts.items():
                                st.write(f"- {prio}: {count}")
                        
                        # Download CSV
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Scarica Risultati CSV",
                            data=csv,
                            file_name="ticket_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Errore nella lettura del file: {e}")
    
    elif page == "📈 Metriche Modelli":
        # Sezione Metriche
        st.header("📈 Metriche di Valutazione Modelli")
        
        try:
            # Carica metriche
            metrics_path = dashboard.models_path / "model_metrics.joblib"
            metrics = joblib.load(metrics_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Modello Categoria")
                st.metric("Accuracy", f"{metrics['category_metrics']['accuracy']:.3f}")
                
                # Classification report
                cat_report = metrics['category_metrics']['classification_report']
                for class_name in ['Tecnico', 'Amministrativo', 'Commerciale']:
                    if class_name in cat_report:
                        precision = cat_report[class_name]['precision']
                        recall = cat_report[class_name]['recall']
                        f1 = cat_report[class_name]['f1-score']
                        st.write(f"**{class_name}:**")
                        st.write(f"- Precision: {precision:.3f}")
                        st.write(f"- Recall: {recall:.3f}")
                        st.write(f"- F1-Score: {f1:.3f}")
            
            with col2:
                st.subheader("📊 Modello Priorità")
                st.metric("Accuracy", f"{metrics['priority_metrics']['accuracy']:.3f}")
                
                # Classification report
                prio_report = metrics['priority_metrics']['classification_report']
                for class_name in ['Bassa', 'Media', 'Alta']:
                    if class_name in prio_report:
                        precision = prio_report[class_name]['precision']
                        recall = prio_report[class_name]['recall']
                        f1 = prio_report[class_name]['f1-score']
                        st.write(f"**{class_name}:**")
                        st.write(f"- Precision: {precision:.3f}")
                        st.write(f"- Recall: {recall:.3f}")
                        st.write(f"- F1-Score: {f1:.3f}")
            
            # Confusion Matrix
            st.subheader("🧩 Matrici di Confusione")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Categoria**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    metrics['category_metrics']['confusion_matrix'],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Tecnico', 'Amministrativo', 'Commerciale'],
                    yticklabels=['Tecnico', 'Amministrativo', 'Commerciale'],
                    ax=ax
                )
                ax.set_title('Confusion Matrix - Categoria')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                st.pyplot(fig)
            
            with col2:
                st.write("**Priorità**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    metrics['priority_metrics']['confusion_matrix'],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Bassa', 'Media', 'Alta'],
                    yticklabels=['Bassa', 'Media', 'Alta'],
                    ax=ax
                )
                ax.set_title('Confusion Matrix - Priorità')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"❌ Impossibile caricare le metriche: {e}")

if __name__ == "__main__":
    main()