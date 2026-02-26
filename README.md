# Vishing-Detection-Prosodic-Model

## Note: 
In Test Mode: Remove All sections for test only before using for full datasets
Extractions

## Folder Structure (Before Running)

Make sure files looks like this:

```
Vishing-Detection-Prosodic-Model/
│
├── Prosodic-Model.ipynb
├── requirements.txt
│
└── ASVspoof5/
    ├── ASVspoof5_protocols.tar.gz
    ├── flac_D_aa.tar
    ├── flac_D_...
    ├── flac_T_aa.tar
    ├── flac_T_...
    ├── flac_E_aa.tar
    └── flac_E_...
```

---

## Running `Prosodic-Model.ipynb`

When you run the notebook, it will automatically:

- Install the required dependencies from `requirements.txt`
- Extract all required `.tar` dataset files
- Extract prosodic features from each audio file
- Cache the extracted features in the `feature_cache/` folder for reuse
- Train the prosodic model
- Save the trained model as `Trained-Prosodic-Model.joblib`

---
## Folder Structure (After Running)

After running, your folder should look like this:

```
Vishing-Detection-Prosodic-Model/
│
├── Prosodic-Model.ipynb
├── requirements.txt
├── Trained-Prosodic-Model.joblib
│
└── ASVspoof5/
    ├── ...
    ├── ASVspoof5_protocols/
    ├── feature_cache/
    ├── flac_D/
    ├── flac_T/
    └── flac_E_eval/
```
