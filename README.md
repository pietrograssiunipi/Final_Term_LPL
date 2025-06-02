# DSL per Grafi – README

## Introduzione

Questo progetto implementa un mini‑linguaggio **imperativo** per la manipolazione di grafi non orientati, modellati come insiemi di archi (A,B).&#x20;

Il linguaggio supporta:

* operazioni sui grafi (`union`, `intersect`, `difference`, `shortest`, `path`);
* dichiarazioni di variabili;
* comandi di controllo di flusso (`if/else`, `while`);
* funzioni definite dall’utente;
* caricamento di grafi da file esterni;
* visualizzazione dei grafi tramite il comando `print`.

Il tutto è scritto in Python 3.11 e si appoggia a **Lark** per il parsing e **networkx / matplotlib** per il disegno.

---

## Requisiti

```bash
pip install lark-parser networkx matplotlib
```

---

## Sintassi essenziale

### Dichiarazioni, assegnamenti, stampa

```minigr
var g  = 'graph1.txt'      # nuova variabile
var h  = 'graph2.txt'
g     <- g union h          # riassegna
print g                     # disegna il grafo
```

### Operazioni sui grafi

| Operatore    | Descrizione                                         |
| ------------ | --------------------------------------------------- |
| `union`      | Unione di archi                                     |
| `intersect`  | Intersezione di archi                               |
| `difference` | Sottrazione degli archi del secondo grafo dal primo |
| `path`       | Un cammino (qualunque) fra due nodi                 |
| `shortest`   | Cammino piú corto fra due nodi                      |

### Controllo di flusso

```minigr
if  7 > 10 then               
    print g
else
    g <- g difference h
endif

var k = 3;
while k > 0 do
    print k;
    k <- k-1
done
```

### Funzioni definite dall’utente

```minigr
function overlay(a,b) = a union b;

var merged = overlay(g,h);
print merged
```

---

## File di input

* **graph1.txt**, **graph2.txt, graph4.txt** – liste di archi `u,v`.
* **pair.txt** – un singolo arco `u,v` che funge da coppia di nodi di interesse (es. per `shortest`).

> ⚠️ Se il codice viene eseguito in locale, si prega di scaricare anche i file dei grafi (.txt) per una corretta esecuzione.
---

## Esempi completi

### 1. Unione di due grafi

```minigr
var g1 = 'graph1.txt';
var g2 = 'graph2.txt';
var g_union = g1 union g2;
print g_union
```

### 2. Scelta condizionale di un grafo

```minigr
var x = 2;
var g = 'graph1.txt';
if x > 1 then
    g <- 'graph1.txt'
else
    g <- 'graph2.txt'
endif;
print g
```

### 3. Costruzione iterativa di percorsi disgiunti

Calcola fino a **tre** cammini a lunghezza minima fra due endpoint, rimuovendoli via via dal grafo di lavoro.

```minigr
var g_work   = 'graph1.txt';
var endpoints = 'pair.txt';      # es. "A,Y"
var result = g_work difference g_work;  # grafo vuoto
var k = 3;
while k > 0 do
    var sp   = endpoints shortest g_work;
    result   <- result union sp;
    g_work   <- g_work difference sp;
    k        <- k - 1
done;
print result
```

### 4. Overlay (funzione utente)

```minigr
function overlay(a,b) = a union b;
var merged = overlay('graph1.txt', 'graph2.txt');
print merged
```

### 5. Funzione con operazioni tra grafi

```minigr
function cap(g,h) = g intersect h;
var g1 = 'graph1.txt';
var g2 = 'graph4.txt';
var result = cap(g1, g2);
print result
```
