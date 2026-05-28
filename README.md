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

Pre inštaláciu základných knižníc použite príkazy (z projektu `pg450oc-main`):

```powershell
# Vytvorenie virtuálneho prostredia
python -m venv .venv

# Aktivácia virtuálneho prostredia
.\.venv\Scripts\Activate.ps1

# Upgrade pip a inštalácia závislostí
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r models/requirements.txt
```

> **Poznámka:** V PowerShell použite vždy `python -m pip` namiesto samotného `pip`.
> Ak používate iný shell alebo OS, nahraďte `Activate.ps1` zodpovedajúcim príkazom (napr. `source .venv/bin/activate` na Linuxe/Mac).

## Popis skriptov

- **CovidGAN/covidgan_testing2.py** — skript pre trénovanie CovidGAN modelu, generovanie náhodných a vyvážených syntetických Covid dát, uloženie výsledkov a vyhodnotenie kvality.
- **CTGAN/ctgan_testing.py** — skript pre trénovanie CTGAN a generovanie syntetických tabulárnych dát.
- **MedGAN/medgan_testing3.py** — skript pre trénovanie MedGAN a vyhodnotenie generovaných dát.
- **WGAN/wgan_testing3.py** — skript pre trénovanie WGAN a generovanie syntetických medicínskych dát.
- **DigitalTwin/digital_twin.py** — skript pre simuláciu digitálneho dvojčaťa a analýzu výsledkov.

## Zoznam použitých modelov

- CovidGAN (ACGAN / podmienená generácia)
- CTGAN (tabulárna GAN generácia)
- MedGAN (medicínsky GAN pre tabulárne dáta)
- WGAN (Wasserstein GAN)
- Digitálne dvojča (Digital Twin) simulácia

## Dôležité poznámky

- **Virtuálne prostredie:** Vždy aktivujte virtuálne prostredie pred spustením skriptov: `.\.venv\Scripts\Activate.ps1`
- **PowerShell:** V PowerShell na Windows vždy použite `python -m pip` namiesto samotného `pip` príkazu.
- **Spustenie skriptov:** Skripty sa môžu spúšťať z priečinku projektu alebo z priečinku konkrétneho modulu.

## Spustenie jednotlivých modulov

**Prerequisites:** Virtuálne prostredie musí byť aktivované (`(.venv)` prefix v príkazovom riadku).

### CovidGAN

```powershell
cd pg450oc-main\CovidGAN
python covidgan_testing2.py
```

### CTGAN

```powershell
cd pg450oc-main\CTGAN
python ctgan_testing.py
```

### MedGAN

```powershell
cd pg450oc-main\MedGAN
python medgan_testing3.py
```

### WGAN

```powershell
cd pg450oc-main\WGAN
python wgan_testing3.py
```

### Digital Twin

```powershell
cd pg450oc-main\DigitalTwin
python digital_twin.py
```

**Alternatívne:** Spustenie z koreňového adresára projektu:

```powershell
python pg450oc-main\CovidGAN\covidgan_testing2.py
python pg450oc-main\CTGAN\ctgan_testing.py
python pg450oc-main\MedGAN\medgan_testing3.py
python pg450oc-main\WGAN\wgan_testing3.py
python pg450oc-main\DigitalTwin\digital_twin.py
```

## Výstupy

Po spustení skriptov sa môžu uložiť:

- syntetické CSV súbory
- trénovacie checkpointy
- grafy a vizualizácie
- výsledky simulácie digitálneho dvojčaťa
