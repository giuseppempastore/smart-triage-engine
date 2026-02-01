"""
Dataset Generator - Triage Automatico dei Ticket
Genera dataset sintetico con ticket di esempio per categorie:
Tecnico, Amministrativo, Commerciale
"""

import csv
import random
from pathlib import Path

# Dataset di template per ogni categoria
TECHNICAL_TITLES = [
    "Server non raggiungibile",
    "Errore database",
    "Applicazione crash",
    "Performance degrado",
    "Problema connessione",
    "Bug interfaccia",
    "Timeout API",
    "Errore autenticazione",
    "File corrotto",
    "Sincronizzazione fallita",
]

TECHNICAL_BODIES = [
    "Il server è irraggiungibile da questa mattina. Non riesco a connettermi.",
    "L'applicazione si chiude inaspettatamente quando accedo al modulo reporting.",
    "La velocità di caricamento è molto lenta, circa 5 minuti per query semplici.",
    "Ricevo errore 500 quando trato di salvare il documento.",
    "La sincronizzazione con il database è fallita alle 14:30.",
    "L'interfaccia non risponde ai click, blocco totale.",
    "Errore di connessione al servizio esterno API payments.",
    "Password non accettata, errore autenticazione",
    "Il backup di questa notte non è stato completato.",
    "Problema certificato SSL scaduto",
]

ADMIN_TITLES = [
    "Fattura non pagata",
    "Errore contabilità",
    "Rimborso richiesto",
    "Documento fiscale mancante",
    "Pagamento duplicato",
    "Richiesta note credito",
    "Discrepanza bilancio",
    "Modifica dati anagrafi",
    "Allegato documentazione",
    "Verifica tassa",
]

ADMIN_BODIES = [
    "La fattura FT2024001 non risulta pagata in tesoreria. Controllare il versamento.",
    "Richiedo rimborso per l'ordine ORD-2024-456 già annullato.",
    "Ho pagato due volte la fattura FT2024002. Necessario rimborso.",
    "Mancano i documenti fiscali per la pratica PA-2024-123.",
    "La documentazione aziendale necessita aggiornamento dati sede.",
    "Verificare il calcolo IVA sulla fattura passiva FT-PASS-789.",
    "Allegato contratto per la verifica approvazione.",
    "Segnalazione discrepanza nel bilancio trimestrale Q1 2024.",
    "Richiedo nota credito per reso merce non conforme.",
    "Dati fornitore inesatti nel sistema.",
]

COMMERCIAL_TITLES = [
    "Richiesta preventivo",
    "Ordine urgente",
    "Offerta speciale",
    "Informazioni prodotto",
    "Problema consegna",
    "Sconto cliente",
    "Informazioni catalogo",
    "Ordine modificazione",
    "Nuovo cliente registrazione",
    "Partnership proposta",
]

COMMERCIAL_BODIES = [
    "Mi serve un preventivo urgente per 100 licenze software. Timeline: entro venerdì.",
    "Ho ricevuto l'ordine ORD-2024-789 ma la quantità è sbagliata.",
    "Vorrei conoscere i termini di sconto per acquisti bulk.",
    "Quali sono le caratteristiche tecniche del prodotto PRD-456?",
    "La consegna dell'ordine ORD-2024-654 era prevista per lunedì ma non è arrivata.",
    "Sono cliente di lunga data, potete fare uno sconto su questo ordine?",
    "Mi interessa sapere se avete un catalogo completo in PDF.",
    "Voglio modificare la quantità nell'ordine ORD-2024-111 prima della spedizione.",
    "Registrazione nuovo cliente per primo ordine di prova.",
    "Proposizione collaborazione per distribuzione esclusiva.",
]

# Parole chiave per la priorità
HIGH_PRIORITY_KEYWORDS = [
    "urgente", "bloccante", "critico", "crash", "errore", "non funziona",
    "down", "irraggiungibile", "emergenza", "subito", "immediatamente",
    "fattura", "pagamento", "pagato", "denaro", "soldi", "rimborso"
]

MEDIUM_PRIORITY_KEYWORDS = [
    "lento", "problema", "discrepanza", "inesatto", "verificare",
    "controllare", "preventivo", "ordine", "consegna", "modifica"
]

def generate_priority(text):
    """Assegna priorità in base a parole chiave nel testo"""
    text_lower = text.lower()
    
    high_count = sum(1 for keyword in HIGH_PRIORITY_KEYWORDS if keyword in text_lower)
    medium_count = sum(1 for keyword in MEDIUM_PRIORITY_KEYWORDS if keyword in text_lower)
    
    if high_count > 0:
        return "Alta"
    elif medium_count > 0:
        return "Media"
    else:
        return "Bassa"

def generate_dataset(output_path, num_tickets=300):
    """
    Genera dataset sintetico di ticket
    
    Args:
        output_path: Percorso dove salvare il CSV
        num_tickets: Numero di ticket da generare
    """
    
    # Dati raggruppati per categoria
    categories_data = {
        "Tecnico": (TECHNICAL_TITLES, TECHNICAL_BODIES),
        "Amministrativo": (ADMIN_TITLES, ADMIN_BODIES),
        "Commerciale": (COMMERCIAL_TITLES, COMMERCIAL_BODIES),
    }
    
    tickets = []
    tickets_per_category = num_tickets // 3
    ticket_id = 1
    
    # Genera ticket per ogni categoria
    for category, (titles, bodies) in categories_data.items():
        for _ in range(tickets_per_category):
            title = random.choice(titles)
            body = random.choice(bodies)
            
            # Assegna priorità in base a parole chiave
            priority = generate_priority(title + " " + body)
            
            tickets.append({
                "id": ticket_id,
                "title": title,
                "body": body,
                "category": category,
                "priority": priority,
            })
            ticket_id += 1
    
    # Aggiungi ticket aggiuntivi per raggiungere il numero desiderato
    remaining = num_tickets - len(tickets)
    for _ in range(remaining):
        category = random.choice(list(categories_data.keys()))
        titles, bodies = categories_data[category]
        title = random.choice(titles)
        body = random.choice(bodies)
        priority = generate_priority(title + " " + body)
        
        tickets.append({
            "id": ticket_id,
            "title": title,
            "body": body,
            "category": category,
            "priority": priority,
        })
        ticket_id += 1
    
    # Mescola i ticket
    random.shuffle(tickets)
    
    # Salva in CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "body", "category", "priority"])
        writer.writeheader()
        writer.writerows(tickets)
    
    print(f"✓ Dataset generato: {output_file}")
    print(f"  Total tickets: {len(tickets)}")
    for category in categories_data.keys():
        count = sum(1 for t in tickets if t["category"] == category)
        print(f"  {category}: {count}")
    
    return output_file

if __name__ == "__main__":
    # Genera dataset sintetico nella cartella data/
    output_path = Path(__file__).parent.parent / "data" / "tickets_dataset.csv"
    generate_dataset(output_path, num_tickets=300)
