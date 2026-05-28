# Modelovanie syntetických medicínskych dát s využitím generatívnych modelov

## Systémová príručka

Tento repozitár obsahuje experimenty s generatívnymi sieťami pre syntetické medicínske dáta a digitálne dvojča. Hlavné moduly sú CovidGAN, CTGAN, MedGAN, WGAN a DigitalTwin. Každý modul má samostatný skript pre načítanie dát, trénovanie/generovanie a vyhodnotenie.

## Použité súčasti

Základné komponenty projektu:

- Python 3.11 alebo novší
- NumPy
- Pandas
- PyTorch
- scikit-learn
- SciPy
- Matplotlib
- Seaborn
- OpenPyXL
- CTGAN knižnica (pre CTGAN modul)

Pre inštaláciu základných knižníc použite príkaz:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> Ak používate iný shell alebo OS, nahraďte `Activate.ps1` zodpovedajúcim príkazom.

## Popis skriptov

- CovidGAN/covidgan_testing2.py — skript pre trénovanie CovidGAN modelu, generovanie náhodných a vyvážených syntetických Covid dát, uloženie výsledkov a vyhodnotenie kvality.
- CTGAN/ctgan_testing.py — skript pre trénovanie CTGAN a generovanie syntetických tabulárnych dát.
- MedGAN/medgan_testing.py — skript pre trénovanie MedGAN a vyhodnotenie generovaných dát.
- WGAN/wgan_testing.py — skript pre trénovanie WGAN a generovanie syntetických medicínskych dát.
- DigitalTwin/digital_twin.py — skript pre simuláciu digitálneho dvojčaťa a analýzu výsledkov.

## Zoznam použitých modelov

- CovidGAN (ACGAN / podmienená generácia)
- CTGAN (tabulárna GAN generácia)
- MedGAN (medicínsky GAN pre tabulárne dáta)
- WGAN (Wasserstein GAN)
- Digitálne dvojča (Digital Twin) simulácia

## Dôležité poznámky

- Skripty je najlepšie spúšťať z adresára projektu alebo z adresára, v ktorom sa nachádzajú príslušné skripty.
- Upravte cesty k datasetom priamo v skriptoch, ak nie sú umiestnené v predpokladanom priečinku datasets/.
- Skontrolujte, či dátové súbory majú správny formát (CSV, XLSX) a očakávané stĺpce.
- Pre trénovanie veľkých modelov je odporúčané použitie GPU.
- Ak používate Windows, spúšťajte python/py v aktívnom virtuálnom prostredí.

## Spustenie jednotlivých modulov

### CovidGAN

```powershell
py covidgan_testing.py
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

### Digital Twin

```powershell
py digital_twin.py
```

## Výstupy

Po spustení skriptov sa môžu uložiť:

- syntetické CSV súbory
- trénovacie checkpointy
- grafy a vizualizácie
- výsledky simulácie digitálneho dvojčaťa
