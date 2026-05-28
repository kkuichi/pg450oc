# Medical Synthetic Data GANs

Tento repozitár obsahuje experimenty s generatívnymi sieťami pre syntetické medicínske dáta. Projekt zahŕňa rôzne GAN architektúry pre generovanie a vyhodnocovanie syntetických Covid dát, vrátane:

- `CovidGAN` 
- `CTGAN` 
- `MedGAN`
- `WGAN`
- `DigitalTwin`


## Štruktúra projektu

- `CovidGAN/` — skripty a výstupy pre CovidGAN
- `CTGAN/` — skripty pre CTGAN experimenty
- `MedGAN/` — skripty pre MedGAN experimenty
- `WGAN/` — skripty pre WGAN experimenty
- `DigitalTwin/` — skripty pre digitálne dvojča a výsledky simulácie
- `datasets/` — vstupné datasety (CSV/XLSX)

## Požiadavky

Odporúčané prostredie:

- Python 3.10+ alebo 3.11
- Virtuálne prostredie `venv`

Inštalácia závislostí (príklad):

```powershell
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

## Spustenie jednotlivých modelov

### CovidGAN

```powershell
py covidgan_testing2.py
```

### CTGAN

```powershell
py ctgan_testing.py
```

### MedGAN

```powershell
py medgan_testing.py
```

### WGAN

```powershell
py wgan_testing.py
```

### DigitalTwin

```powershell
py digital_twin.py
```

## Výstupy

Po spustení skriptov sa môžu uložiť:

- syntetické CSV súbory
- trénovacie checkpointy
- grafy a vizualizácie

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
