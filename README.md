# Medical Synthetic Data GANs

Tento repozitár obsahuje experimenty s generatívnymi sieťami pre syntetické medicínske dáta. Projekt zahŕňa rôzne GAN architektúry pre generovanie a vyhodnocovanie syntetických Covid dát, vrátane:

- `CovidGAN` (ACGAN-style generátor pre podmienené dáta)
- `CTGAN` (tabulárna GAN generácia)
- `MedGAN`
- `WGAN`
- `DigitalTwin` (digitálny dvojča model pre simuláciu)
- Porovnávacie skripty a metriky

## Štruktúra projektu

- `CovidGAN/` — skripty a výstupy pre CovidGAN
- `CTGAN/` — skripty pre CTGAN experimenty
- `MedGAN/` — skripty pre MedGAN experimenty
- `WGAN/` — skripty pre WGAN experimenty
- `DigitalTwin/` — skripty pre digitálne dvojča a výsledky simulácie
- `datasets/` — vstupné datasety (CSV/XLSX)
- `comparison_results/` — výsledné tabuľky porovnaní
- `compare_models.py` — porovnanie modelov na jednom datasete
- `compare_models_multiple_datasets.py` — porovnanie cez viaceré datasety

## Požiadavky

Odporúčané prostredie:

- Python 3.10+ alebo 3.11
- Virtuálne prostredie `venv`

Inštalácia závislostí (príklad):

```powershell
cd C:\Users\pepin\Desktop\bakalarka_modely
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch scikit-learn scipy matplotlib seaborn pandas openpyxl ctgan
```

> Ak používate iný systém, nahraďte `Activate.ps1` za `activate` (Windows CMD) alebo `source .venv/bin/activate` (Linux/macOS).

## Dáta

Skripty hľadajú datasetové súbory v priečinku `datasets/`. Podporované formáty sú:

- `.csv`
- `.xls`
- `.xlsx`

Uistite sa, že názvy stĺpcov v datasete zodpovedajú premenným v skriptoch, napríklad `TARGET_COL` v `CovidGAN/covidgan_testing2.py`.

## Spustenie jednotlivých modelov

### CovidGAN

```powershell
cd C:\Users\pepin\Desktop\bakalarka_modely\CovidGAN
py covidgan_testing2.py
```

Tento skript:

- načíta dataset(y) z koreňového `datasets/`
- predspracuje dáta
- natrénuje CovidGAN
- vygeneruje náhodné aj vyvážené syntetické dáta
- uloží výsledné CSV súbory a obrázky do priečinka `CovidGAN`

### CTGAN

```powershell
cd C:\Users\pepin\Desktop\bakalarka_modely\CTGAN
py ctgan_testing.py
```

### MedGAN

```powershell
cd C:\Users\pepin\Desktop\bakalarka_modely\MedGAN
py medgan_testing.py
```

### WGAN

```powershell
cd C:\Users\pepin\Desktop\bakalarka_modely\WGAN
py wgan_testing.py
```

### DigitalTwin

```powershell
cd C:\Users\pepin\Desktop\bakalarka_modely\DigitalTwin
py digital_twin.py
```

## Porovnávacie skripty

Pre porovnanie viac modelov alebo datasetov použite:

```powershell
cd C:\Users\pepin\Desktop\bakalarka_modely
py compare_models.py
py compare_models_multiple_datasets.py
```

Tieto skripty vytvárajú výsledné súbory v priečinku `comparison_results/`.

## Výstupy

Po spustení skriptov sa môžu uložiť:

- syntetické CSV súbory
- trénovacie checkpointy
- grafy a vizualizácie
- porovnávacie tabuľky

## Upraviteľné nastavenia

Každý skript obsahuje sekciu konfigurácie, kde môžete upraviť:

- cesty k datasetom
- názov cieľového stĺpca
- hyperparametre modelu
- počet epôch a batch veľkosť
- cesty pre ukladanie checkpointov

## Tipy

- Pred každým spustením odporúčam aktivovať virtuálne prostredie.
- Datasety uložte do `datasets/` a skontrolujte názvy stĺpcov.
- Ak používate GPU, skripty automaticky využijú PyTorch zariadenie `cuda`, ak je dostupné.

## Licencia

Pridajte si svoju vlastnú licenciu podľa potreby, napríklad `MIT License` alebo inú vhodnú licenciu pre vašu prácu.
