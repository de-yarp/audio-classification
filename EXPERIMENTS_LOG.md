# Protokol experimentov: Optimalizácia CNN na Mel-spektrogramoch (ESC-50)

Tento dokument sumarizuje priebeh experimentov, hypotézy a výsledky dosiahnuté pri vývoji klasifikátora zvukových udalostí na báze Mel-spektrogramov.

---

## Cieľ experimentov
Prekonať základnú úroveň presnosti (baseline) nastavenú na 50 % (MFCC model) a navrhnúť architektúru CNN schopnú efektívne extrahovať príznaky z Mel-spektrogramov.

---

## Experiment č. 1: Baseline Mel-spektrogram
- **Konfigurácia:** `cnn_mel_1.yaml`
- **Architektúra:** 2 konvolučné bloky [32, 64], jadro 3x3, 1 FC vrstva [256].
- **Hypotéza:** Mel-spektrogramy poskytnú bohatší frekvenčný kontext ako MFCC.
- **Výsledok:** **44.25 % accuracy.**
- **Pozorovanie:** Model je poddimenzovaný (underfitted). Kapacita siete nepostačuje na rozlíšenie 50 komplexných tried zvukov.

## Experiment č. 2: Zvýšenie kapacity siete
- **Konfigurácia:** `cnn_mel_2.yaml`
- **Zmeny:** Pridaný 3. konvolučný blok (128 filtrov), FC vrstva rozšírená na 512 neurónov, zvýšený počet epoch na 100.
- **Výsledok:** **~65 % accuracy.**
- **Záver:** Hypotéza potvrdená. Vyššia hĺbka siete umožnila lepšiu abstrakciu akustických príznakov.

## Experiment č. 3: Hĺbka a SpecAugment (Regularizácia)
- **Konfigurácia:** `cnn_mel_3.yaml`
- **Zmeny:** 4 konvolučné bloky (do 256 filtrov), implementácia SpecAugment (maskovanie frekvencií a času), LR Warmup.
- **Výsledok:** **61.25 % accuracy.**
- **Pozorovanie:** Pokles presnosti. Model sa začal preučovať (overfitting) na malom datasete. Augmentácia bola príliš agresívna (maskovanie > 60 % signálu), čo sťažilo proces učenia.

## Experiment č. 4: Finálny "Monster" Tuning (Víťazný model)
- **Konfigurácia:** `cnn_mel_4.yaml`
- **Kľúčové zmeny:**
    1. **Jadro 5x5 v prvej vrstve:** Lepšie zachytenie vertikálnych frekvenčných závislostí (harmonických zložiek) hneď na vstupe.
    2. **ADAMW + Weight Decay (0.001):** Použitie modernejšej verzie optimizéra pre lepšiu regularizáciu váh.
    3. **Cosine Annealing Scheduler:** Plynulé znižovanie Learning Rate podľa sínusoidy pre stabilnejšiu konvergenciu.
    4. **Optimalizovaný SpecAugment:** Redukcia počtu masiek na 3 pre zachovanie kritickej informácie.
- **Výsledok:** **74.5 % accuracy.**
- **Záver:** Dosiahnutá optimálna rovnováha medzi hĺbkou siete, šírkou konvolučných jadier a regularizáciou. Tento model výrazne prekonáva baseline o takmer 25 %.

---

## Technické zhrnutie
Všetky výsledky sú podrobne logované v súbore `runs/cnn_mel_train_tracker.csv`. Najlepší model sa nachádza v príslušnom artifact priečinku pre run ID prislúchajúci experimentu č. 4.
