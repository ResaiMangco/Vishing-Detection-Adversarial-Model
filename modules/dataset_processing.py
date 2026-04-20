from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
import tarfile
from typing import Dict, Iterable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf


PROTOCOL_COLUMNS = [
    "SPEAKER_ID",
    "FLAC_FILE_NAME",
    "SPEAKER_GENDER",
    "CODEC",
    "CODEC_Q",
    "CODEC_SEED",
    "ATTACK_TAG",
    "ATTACK_LABEL",
    "KEY",
    "TMP",
]

SPLIT_TO_PROTOCOL_FILE: Dict[str, str] = {
    "train": "ASVspoof5.train.tsv",
    "dev": "ASVspoof5.dev.track_1.tsv",
    "eval": "ASVspoof5.eval.track_1.tsv",
}

SPLIT_TO_AUDIO_DIR: Dict[str, str] = {
    "train": "flac_T",
    "dev": "flac_D",
    "eval": "flac_E_eval",
}

SPLIT_TO_FILE_PREFIX: Dict[str, str] = {
    "train": "T_",
    "dev": "D_",
    "eval": "E_",
}

SPLIT_TO_ARCHIVE_PREFIX: Dict[str, str] = {
    "train": "flac_T_",
    "dev": "flac_D_",
    "eval": "flac_E_",
}


@dataclass(frozen=True)
class DataLayout:
    data_root: Path
    protocol_root: Path
    train_audio_root: Path
    dev_audio_root: Path
    eval_audio_root: Path

@dataclass(frozen=True)
class WaveformSample:
    waveform: torch.Tensor
    label: torch.Tensor
    file_name: str
    file_path: str
    split: str
    attack_label: str
    key: str

class WaveformDataset(Dataset[Dict[str, object]]):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        sample_rate: int = 16000,
        target_num_samples: Optional[int] = None,
        random_crop: bool = False,
        normalize_waveform: bool = True,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.sample_rate = sample_rate
        self.target_num_samples = target_num_samples
        self.random_crop = random_crop
        self.normalize_waveform = normalize_waveform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.dataframe.iloc[index]
        label_value = int(row["LABEL"]) if "LABEL" in row else map_binary_label(row["KEY"])
        waveform = self._load_waveform(str(row["FULL_FILE_PATH"]))

        return {
            "waveform": waveform,
            "label": torch.tensor(label_value, dtype=torch.long),
            "file_name": str(row["FLAC_FILE_NAME"]),
            "file_path": str(row["FULL_FILE_PATH"]),
            "split": str(row.get("SPLIT", "unknown")),
            "attack_label": str(row.get("ATTACK_LABEL", "")),
            "key": str(row.get("KEY", "")),
        }

    def _load_waveform(self, file_path: str) -> torch.Tensor:
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except (RuntimeError, OSError, ImportError):
            # TorchCodec can fail when FFmpeg shared libraries are unavailable.
            audio_array, sample_rate = sf.read(file_path, always_2d=True, dtype="float32")
            waveform = torch.from_numpy(audio_array.T)

        waveform = waveform.to(torch.float32)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )

        waveform = waveform.squeeze(0)
        if self.normalize_waveform:
            peak = waveform.abs().max().clamp_min(1e-6)
            waveform = waveform / peak

        waveform = modify_waveform(
            waveform=waveform,
            target_num_samples=self.target_num_samples,
            random_crop=self.random_crop,
        )
        
        return waveform


def resolve_layout(data_root: str | Path) -> DataLayout:
    """Returns the dataset layout based on the provided data root directory."""
    root = Path(data_root).expanduser().resolve()
    protocol_root = root / "ASVspoof5_protocols"
    if not protocol_root.exists():
        protocol_root = root

    return DataLayout(
        data_root=root,
        protocol_root=protocol_root,
        train_audio_root=root / SPLIT_TO_AUDIO_DIR["train"],
        dev_audio_root=root / SPLIT_TO_AUDIO_DIR["dev"],
        eval_audio_root=root / SPLIT_TO_AUDIO_DIR["eval"],
    )

def extract_archives(data_root: str | Path, max_items: Optional[int] = None, balanced: bool = False) -> None:
    """
        Extracts dataset archives, optionally limiting extracted audio items.

        When ``max_items`` is provided, extraction of split audio archives stops after
        that many archive members have been extracted in total.

        When ``balanced`` is True and ``max_items`` is set, extraction uses the
        protocol TSV to identify bonafide vs spoof files and extracts up to
        ``max_items // 2`` of each class so the on-disk split is class-balanced.
    """
    layout = resolve_layout(data_root)

    protocol_archive = layout.data_root / "ASVspoof5_protocols.tar.gz"
    if protocol_archive.exists() and not (layout.data_root / "ASVspoof5_protocols").exists():
        with tarfile.open(protocol_archive, mode="r:gz") as archive_handle:
            archive_handle.extractall(path=layout.data_root)

    for split, archive_prefix in SPLIT_TO_ARCHIVE_PREFIX.items():
        target_dir = layout.data_root / SPLIT_TO_AUDIO_DIR[split]
        print(target_dir)
        if target_dir.exists():
            print(f"{target_dir} already exists.")
            continue

        archive_paths = sorted(layout.data_root.glob(f"{archive_prefix}*.tar"))
        print(archive_paths)
        if not archive_paths:
            print(f"No archive paths found for split {split}")
            continue

        if balanced and max_items is not None:
            protocol_df = load_asvspoof_tsv(get_protocol_path(data_root=layout.data_root, split=split))
            label_source = "KEY" if "KEY" in protocol_df.columns else "ATTACK_LABEL"
            bonafide_stems = set(
                protocol_df.loc[
                    protocol_df[label_source].str.strip().str.lower() == "bonafide",
                    "FLAC_FILE_NAME",
                ]
            )

            per_class_limit = max_items // 2
            bonafide_remaining = per_class_limit
            spoof_remaining = per_class_limit

            for archive_path in archive_paths:
                if bonafide_remaining <= 0 and spoof_remaining <= 0:
                    break

                with tarfile.open(archive_path, mode="r") as archive_handle:
                    members = archive_handle.getmembers()
                    random.shuffle(members)
                    selected = []
                    for m in members:
                        if not m.isfile():
                            selected.append(m)
                            continue
                        stem = Path(m.name).stem
                        if stem in bonafide_stems:
                            if bonafide_remaining > 0:
                                selected.append(m)
                                bonafide_remaining -= 1
                        else:
                            if spoof_remaining > 0:
                                selected.append(m)
                                spoof_remaining -= 1
                        if bonafide_remaining <= 0 and spoof_remaining <= 0:
                            break
                    if selected:
                        archive_handle.extractall(path=layout.data_root, members=selected)
        else:
            remaining_items = max_items
            for archive_path in archive_paths:
                if remaining_items is not None and remaining_items <= 0:
                    break

                with tarfile.open(archive_path, mode="r") as archive_handle:
                    if remaining_items is None:
                        archive_handle.extractall(path=layout.data_root)
                        continue

                    members = archive_handle.getmembers()
                    random.shuffle(members)
                    selected_members = members[:remaining_items]
                    if not selected_members:
                        continue

                    archive_handle.extractall(path=layout.data_root, members=selected_members)
                    remaining_items -= len(selected_members)

def load_asvspoof_tsv(tsv_path: str | Path) -> pd.DataFrame:
    """Loads a protocol TSV file into a DataFrame."""
    return pd.read_csv(
        tsv_path,
        sep=r"\s+",
        names=PROTOCOL_COLUMNS,
        engine="python",
        dtype=str,
    )

def get_protocol_path(data_root: str | Path, split: str) -> Path:
    """Returns the path to the protocol file for a specific split."""
    split = split.lower()
    layout = resolve_layout(data_root)
    protocol_path = layout.protocol_root / SPLIT_TO_PROTOCOL_FILE[split]
    return protocol_path

def get_audio_root(data_root: str | Path, split: str) -> Path:
    """Returns the root directory containing audio files for a specific split."""
    split = split.lower()
    if split not in SPLIT_TO_AUDIO_DIR:
        raise ValueError(f"Unsupported split: {split}")

    layout = resolve_layout(data_root)
    audio_root = getattr(layout, f"{split}_audio_root")
    return audio_root

def build_full_file_path(file_name: str, split: str, data_root: str | Path) -> str:
    """Constructs the full file path for a given audio file name based on the split and data root."""
    split = split.lower()
    expected_prefix = SPLIT_TO_FILE_PREFIX[split]
    if not file_name.startswith(expected_prefix):
        raise ValueError(
            f"Unexpected file name prefix for split '{split}': {file_name}"
        )

    return str(get_audio_root(data_root, split) / f"{file_name}.flac")

def map_binary_label(label_value: str) -> int:
    """Maps the original ASVspoof 5 label value to a binary label (0 for bonafide, 1 for spoof)."""
    label_value = str(label_value).strip().lower()
    return 0 if label_value == "bonafide" else 1

def add_full_file_paths(
    dataframe: pd.DataFrame,
    split: str,
    data_root: str | Path,
) -> pd.DataFrame:
    """Adds full file paths and binary labels to the ASVspoof 5 protocol DataFrame for a specific split."""
    output = dataframe.copy()
    output["SPLIT"] = split.lower()
    output["FULL_FILE_PATH"] = output["FLAC_FILE_NAME"].apply(
        lambda file_name: build_full_file_path(file_name=file_name, split=split, data_root=data_root)
    )

    label_source = "KEY" if "KEY" in output.columns else "ATTACK_LABEL"
    output["LABEL"] = output[label_source].apply(map_binary_label).astype(int)
    return output

def filter_existing_files(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where the full file path exists."""
    file_mask = dataframe["FULL_FILE_PATH"].map(lambda value: Path(value).is_file())
    return dataframe[file_mask].reset_index(drop=True)

def prepare_split_dataframe(
    data_root: str | Path,
    split: str,
    require_existing_files: bool = True,
) -> pd.DataFrame:
    """Prepares a DataFrame for a specific split of the ASVspoof 5 dataset, including full file paths and binary labels."""
    dataframe = load_asvspoof_tsv(get_protocol_path(data_root=data_root, split=split))
    dataframe = add_full_file_paths(dataframe=dataframe, split=split, data_root=data_root)
    if require_existing_files:
        dataframe = filter_existing_files(dataframe)
    return dataframe

def prepare_all_split_dataframes(
    data_root: str | Path,
    require_existing_files: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Prepares dataframes for all splits of the ASVspoof 5 dataset."""
    return {
        split: prepare_split_dataframe(
            data_root=data_root,
            split=split,
            require_existing_files=require_existing_files,
        )
        for split in SPLIT_TO_PROTOCOL_FILE
    }

def subsample_by_class(
    dataframe: pd.DataFrame,
    bonafide_samples: Optional[int] = None,
    spoof_samples: Optional[int] = None,
    random_seed: int = 67,
    balanced: bool = False,
) -> pd.DataFrame:
    """Subsamples a DataFrame to have a specific number of bonafide and spoof samples.

    When balanced=True, both classes are capped to the size of the smaller
    class after initial subsampling so that they are equal in count.
    """
    bonafide_df = dataframe[dataframe["LABEL"] == 0]
    spoof_df = dataframe[dataframe["LABEL"] == 1]

    n_bonafide = min(bonafide_samples, len(bonafide_df)) if bonafide_samples is not None else len(bonafide_df)
    n_spoof = min(spoof_samples, len(spoof_df)) if spoof_samples is not None else len(spoof_df)

    if balanced:
        n_bonafide = n_spoof = min(n_bonafide, n_spoof)

    sampled_frames = [
        bonafide_df.sample(n=n_bonafide, random_state=random_seed),
        spoof_df.sample(n=n_spoof, random_state=random_seed),
    ]

    return (
        pd.concat(sampled_frames, axis=0)
        .sample(frac=1.0, random_state=random_seed + 1)
        .reset_index(drop=True)
    )

def modify_waveform(
    waveform: torch.Tensor,
    target_num_samples: Optional[int],
    random_crop: bool = False,
) -> torch.Tensor:
    """
        Pads or crops a waveform to a target number of samples. If cropping is needed, 
        it can be done randomly or centered.
    """
    if target_num_samples is None:
        return waveform

    current_num_samples = int(waveform.shape[-1])
    if current_num_samples == target_num_samples:
        return waveform

    if current_num_samples > target_num_samples:
        excess = current_num_samples - target_num_samples
        if random_crop and excess > 0:
            start = int(torch.randint(0, excess + 1, (1,)).item())
        else:
            start = excess // 2
        end = start + target_num_samples
        return waveform[..., start:end]

    pad_amount = target_num_samples - current_num_samples
    return torch.nn.functional.pad(waveform, (0, pad_amount))
