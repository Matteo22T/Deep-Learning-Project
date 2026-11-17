# Guida semplice alla CLI di Kaggle

Guida passo passo per usare Kaggle da terminale: installazione, dataset, notebook e problemi comuni.

---

## 1. Installazione

```bash
pip install kaggle
kaggle --version
```

Se il comando non viene trovato, usa `python -m kaggle`.

## 2. Autenticazione

1. Vai su kaggle.com, clicca sulla tua foto profilo e apri **Settings**.
2. Nella sezione **API** clicca **Create New Token**: si scarica `kaggle.json`.
3. Spostalo nella cartella giusta:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

In alternativa, con variabili d'ambiente:

```bash
export KAGGLE_USERNAME=tuo_username
export KAGGLE_KEY=la_tua_chiave
```

> Le variabili d'ambiente hanno la precedenza sul file. Se ricevi errori di permessi strani, controlla con `env | grep KAGGLE` di non avere valori vecchi.

Se il token non funziona, controlla la documentazione ufficiale (github.com/Kaggle/kaggle-api): il formato dei token può cambiare.

## 3. Regola d'oro sui nomi

- Il comando è `kaggle kernels` (con la **s**), non `kernel`.
- Dataset e notebook si identificano con `utente/slug`, ad esempio `matteotocci/impainting-csv`.
- Lo slug **non** è sempre uguale al titolo: copialo da `kaggle kernels list --mine` invece di scriverlo a mano (un errore di battitura dà un falso errore di permessi).

## 4. Dataset

### Scaricare un dataset

```bash
kaggle datasets list -s "titanic"                       # cercare
kaggle datasets files zynicide/wine-reviews              # vedere i file
kaggle datasets download zynicide/wine-reviews --unzip -p ./dati
```

### Creare un tuo dataset

```bash
mkdir mio_dataset
cp *.csv mio_dataset/
kaggle datasets init -p mio_dataset       # crea dataset-metadata.json
```

Modifica `dataset-metadata.json`:

```json
{
  "title": "mio-dataset",
  "id": "tuo_username/mio-dataset",
  "licenses": [{"name": "CC0-1.0"}]
}
```

Poi pubblica:

```bash
kaggle datasets create -p mio_dataset
```

Per aggiornarlo in seguito:

```bash
kaggle datasets version -p mio_dataset -m "Aggiornamento dati"
```

Per vedere i tuoi dataset: `kaggle datasets list --mine`.

## 5. Notebook: eseguirli su Kaggle

### 5.1 Struttura della cartella

Ogni notebook ha la sua cartella con il suo `kernel-metadata.json` (il nome del file deve essere esattamente questo):

```
nb1/
├── 01_notebook.ipynb
└── kernel-metadata.json
```

### 5.2 Il notebook deve avere il kernelspec

Kaggle esegue i notebook con papermill, che richiede `metadata.kernelspec`. Se manca, vedi l'errore `No kernel name found in notebook`. Per aggiungerlo:

```bash
python - <<'EOF'
import json
p = "01_notebook.ipynb"
nb = json.load(open(p))
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
json.dump(nb, open(p, "w"), indent=1)
EOF
```

(Un notebook salvato da Jupyter o VS Code lo ha già.)

### 5.3 `kernel-metadata.json`

Generalo con `kaggle kernels init -p .` e modificalo così:

```json
{
  "id": "tuo_username/nome-notebook",
  "title": "nome-notebook",
  "code_file": "01_notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["tuo_username/mio-dataset"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

Punti importanti:

- Lo slug in `id` deve corrispondere al `title` (minuscolo, con trattini).
- Usa veri booleani (`true`/`false`), non stringhe.
- `dataset_sources` monta i dataset in `/kaggle/input/...`.
- `enable_internet: true` serve per `git clone` e `pip install`.
- `enable_gpu: true` per usare la GPU.

### 5.4 Lanciare, controllare, scaricare

```bash
kaggle kernels push -p .                                  # carica ED esegue
kaggle kernels status tuo_username/nome-notebook          # stato
kaggle kernels output tuo_username/nome-notebook -p ../output_kaggle
```

Stati possibili: `RUNNING`, `COMPLETE`, `ERROR`.

Per aspettare la fine in automatico:

```bash
while true; do
  s=$(kaggle kernels status tuo_username/nome-notebook)
  echo "$s"
  echo "$s" | grep -qiE "complete|error|cancel" && break
  sleep 30
done
```

> Tieni la cartella degli output **fuori** dalla cartella del notebook, altrimenti il prossimo `push` potrebbe caricarla.

## 6. Usare il codice della tua repo nel notebook

Il `push` carica solo i file nella cartella del notebook, non tutta la repo. Per importare moduli tuoi (es. `src/`) puoi clonare la repo dentro il notebook:

```python
import os, sys, subprocess
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()

if ON_KAGGLE:
    repo = Path("/tmp/repo")          # in /tmp: non finisce negli output
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/TUO_UTENTE/TUA_REPO.git", str(repo)],
            check=True,
        )
    sys.path.append(str(repo))

    # trova i dati senza indovinare il percorso
    found = list(Path("/kaggle/input").rglob("nome_file.csv"))
    assert found, "file non trovato in /kaggle/input"
    data_dir = found[0].parent

    os.chdir("/kaggle/working")       # i file salvati con path relativi finiscono negli output
else:
    data_dir = Path("data")
```

Se la repo è privata, salva un token come **Kaggle Secret** dalla UI web (Add-ons, Secrets) e leggilo con:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
```

## 7. Cosa finisce negli output

`kaggle kernels output` scarica **tutto** quello che il notebook scrive in `/kaggle/working`, più il log (`.log`) e l'HTML dei risultati. Quindi:

- salva i risultati (grafici, CSV) in `/kaggle/working`;
- non clonare la repo in `/kaggle/working`, altrimenti la riscarichi intera.

## 8. Leggere gli errori

Se lo stato è `ERROR`:

```bash
kaggle kernels output tuo_username/nome-notebook -p ../output_kaggle
tail -n 60 ../output_kaggle/*.log
grep -iE "error|exception|traceback" ../output_kaggle/*.log | head
```

## 9. Problemi comuni

| Errore | Causa e soluzione |
|---|---|
| `invalid choice: 'kernel'` | Scrivi `kernels` con la s |
| `Permission 'kernels.get' was denied` | Slug sbagliato (controlla con `kernels list --mine`), account diverso o notebook inesistente |
| `No kernel name found in notebook` | Manca `kernelspec` nel `.ipynb` (vedi 5.2) |
| `ModuleNotFoundError: No module named 'src'` | La repo non è sul server: clonala (vedi 6) |
| `FileNotFoundError` su `/kaggle/input/...` | Percorso del dataset diverso: usa `rglob` per trovarlo |
| `401 Unauthorized` | Token errato o scaduto: rigeneralo |
| `403 Forbidden` su una competizione | Accetta prima le regole dal sito |
| `kaggle: command not found` | Usa `python -m kaggle` o aggiungi `~/.local/bin` al PATH |

## 10. Comandi di riferimento rapido

```bash
# Dataset
kaggle datasets list --mine
kaggle datasets download utente/dataset --unzip -p ./dati
kaggle datasets create -p ./cartella
kaggle datasets version -p ./cartella -m "messaggio"

# Notebook
kaggle kernels list --mine
kaggle kernels init -p .
kaggle kernels push -p .
kaggle kernels status utente/slug
kaggle kernels output utente/slug -p ./output
kaggle kernels pull utente/slug -p ./scaricato

# Aiuto
kaggle --help
kaggle kernels --help
```

## 11. Flusso completo in 6 passi

1. Installa e autenticati (`pip install kaggle`, `kaggle.json`).
2. Carica i tuoi dati come dataset (`kaggle datasets create`).
3. Prepara la cartella del notebook con `.ipynb` (con kernelspec) e `kernel-metadata.json`.
4. Lancia con `kaggle kernels push -p .`.
5. Controlla con `kaggle kernels status utente/slug`.
6. Scarica i risultati con `kaggle kernels output utente/slug -p ../output_kaggle`.
