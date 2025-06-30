import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import wandb
from accelerate import cpu_offload
from torch.utils.data import DistributedSampler
from PIL import Image
from typing_extensions import override

from cogkit.finetune.base import BaseTrainer
from cogkit.finetune.samplers import DistPackingSampler
from cogkit.utils import expand_list, guess_generation_mode
from cogkit.types import GenerationMode
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video

from ..utils import (
    free_memory,
    get_memory_statistics,
    gather_object,
    is_main_process,
    mkdir,
)
from .schemas import DiffusionArgs, DiffusionComponents, DiffusionState


class DiffusionTrainer(BaseTrainer):
    """Trainer‑Klasse mit stabilem Validation‑Logging.

    Alle generierten Artefakte (PNG / MP4 / TXT) übernehmen den Dateistamm der
    jeweiligen Eingabe. Falls kein Name im Batch vorhanden ist, wird weiterhin
    auf ``process{rank}-batch{idx}`` zurückgegriffen.
    """

    # ------------------------------------------------------------------
    #                    INITIALISIERUNG & BASIC SETUP
    # ------------------------------------------------------------------
    @override
    def __init__(self, uargs_fpath: str | Path) -> None:
        super().__init__(uargs_fpath)
        self.uargs: DiffusionArgs
        self.state: DiffusionState
        self.components: DiffusionComponents

    @override
    def _init_args(self, uargs_fpath: Path) -> DiffusionArgs:
        return DiffusionArgs.parse_from_yaml(uargs_fpath)

    @override
    def _init_state(self) -> DiffusionState:
        state = DiffusionState(**super()._init_state().model_dump())
        state.train_resolution = self.uargs.train_resolution
        return state

    @override
    def prepare_models(self) -> None:
        if self.components.vae is not None:
            if self.uargs.enable_slicing:
                self.components.vae.enable_slicing()
            if self.uargs.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    # ------------------------------------------------------------------
    #                           DATASET SETUP
    # ------------------------------------------------------------------
    @override
    def prepare_dataset(self) -> None:  # noqa: C901  (cyclomatic complexity)
        generation_mode = guess_generation_mode(self.components.pipeline_cls)
        match generation_mode:
            case GenerationMode.TextToImage:
                from cogkit.finetune.datasets import (
                    T2IDatasetWithFactorResize,
                    T2IDatasetWithPacking,
                    T2IDatasetWithResize,
                )

                dataset_cls = T2IDatasetWithResize
                if self.uargs.enable_packing:
                    dataset_cls = T2IDatasetWithFactorResize
                    dataset_cls_packing = T2IDatasetWithPacking

            case GenerationMode.TextToVideo:
                from cogkit.finetune.datasets import BaseT2VDataset, T2VDatasetWithResize

                dataset_cls = T2VDatasetWithResize
                if self.uargs.enable_packing:
                    dataset_cls = BaseT2VDataset
                    raise NotImplementedError("Packing for T2V is not implemented")

            case GenerationMode.ImageToVideo:
                from cogkit.finetune.datasets import BaseI2VDataset, I2VDatasetWithResize

                dataset_cls = I2VDatasetWithResize
                if self.uargs.enable_packing:
                    dataset_cls = BaseI2VDataset
                    raise NotImplementedError("Packing for I2V is not implemented")

            case _:
                raise ValueError(f"Invalid generation mode: {generation_mode}")

        additional_args: dict[str, Any] = {
            "device": self.state.device,
            "trainer": self,
        }

        # -------------------- Datasets --------------------
        self.train_dataset = dataset_cls(
            **self.uargs.model_dump(),
            **additional_args,
            using_train=True,
        )
        if self.uargs.do_validation:
            self.test_dataset = dataset_cls(
                **self.uargs.model_dump(),
                **additional_args,
                using_train=False,
            )

        # ---------------- Encode‑Prep --------------------
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae.to(self.state.device, dtype=self.state.weight_dtype)
        if self.uargs.low_vram:
            cpu_offload(self.components.text_encoder, self.state.device)
        else:
            self.components.text_encoder.to(self.state.device, dtype=self.state.weight_dtype)

        # Vorberechnete Embeddings
        self.logger.info("Precomputing embedding ...")
        self.state.negative_prompt_embeds = self.get_negtive_prompt_embeds().to(self.state.device)

        for dataset in (self.train_dataset, self.test_dataset):
            if dataset is None:
                continue
            tmp_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=self.collate_fn,
                batch_size=1,
                num_workers=0,
                pin_memory=self.uargs.pin_memory,
                sampler=DistributedSampler(
                    dataset,
                    num_replicas=self.state.world_size,
                    rank=self.state.global_rank,
                    shuffle=False,
                ),
            )
            for _ in tmp_loader:
                pass

        self.logger.info("Precomputing embedding ... Done")
        dist.barrier()

        self.components.vae = self.components.vae.to("cpu")
        if not self.uargs.low_vram:
            self.components.text_encoder = self.components.text_encoder.to("cpu")
        free_memory()

        # -------------------- DataLoader --------------------
        if not self.uargs.enable_packing:
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                collate_fn=self.collate_fn,
                batch_size=self.uargs.batch_size,
                num_workers=self.uargs.num_workers,
                pin_memory=self.uargs.pin_memory,
                sampler=DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.state.world_size,
                    rank=self.state.global_rank,
                    shuffle=True,
                ),
            )
        else:
            length_list = [self.sample_to_length(s) for s in self.train_dataset]
            self.train_dataset = dataset_cls_packing(self.train_dataset)
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                collate_fn=self.collate_fn_packing,
                batch_size=self.uargs.batch_size,
                num_workers=self.uargs.num_workers,
                pin_memory=self.uargs.pin_memory,
                sampler=DistPackingSampler(
                    length_list,
                    self.state.training_seq_length,
                    shuffle=True,
                    world_size=self.state.world_size,
                    global_rank=self.state.global_rank,
                ),
            )

        if self.uargs.do_validation:
            self.test_data_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                collate_fn=self.collate_fn,
                batch_size=1,
                num_workers=self.uargs.num_workers,
                pin_memory=self.uargs.pin_memory,
                sampler=DistributedSampler(
                    self.test_dataset,
                    num_replicas=self.state.world_size,
                    rank=self.state.global_rank,
                    shuffle=False,
                ),
            )

    # ------------------------------------------------------------------
    #                              VALIDIERUNG
    # ------------------------------------------------------------------
    @override
    def validate(self, step: int, ckpt_path: str | None = None) -> None:  # noqa: C901
        self.logger.info("Starting validation")
        if len(self.test_data_loader) == 0:
            self.logger.warning("No validation samples found. Skipping validation.")
            return

        torch.set_grad_enabled(False)
        self.logger.info(
            "Memory before validation start: "
            + json.dumps(get_memory_statistics(self.logger), indent=4)
        )

        pipe = self.initialize_pipeline(ckpt_path=ckpt_path)
        if self.uargs.low_vram:
            pipe.enable_sequential_cpu_offload(device=self.state.device)
        else:
            pipe.enable_model_cpu_offload(device=self.state.device)
        pipe = pipe.to(dtype=self.state.weight_dtype)

        all_processes_artifacts: list[dict[str, Any]] = []
        for i, batch in enumerate(self.test_data_loader):
            prompt = (batch.get("prompt") or [None])[0]
            prompt_embedding = batch.get("prompt_embedding")

            image = (batch.get("image") or [None])[0]
            encoded_image = batch.get("encoded_image")

            video = (batch.get("video") or [None])[0]
            encoded_video = batch.get("encoded_video")

            self.logger.debug(
                "Validating sample %d/%d on processed %d. Prompt: %s",
                i + 1,
                len(self.test_data_loader),
                self.state.global_rank,
                prompt,
            )

            val_res = self.validation_step(
                pipe=pipe,
                eval_data={
                    "prompt": prompt,
                    "prompt_embedding": prompt_embedding,
                    "image": image,
                    "encoded_image": encoded_image,
                    "video": video,
                    "encoded_video": encoded_video,
                },
            )

            artifacts: dict[str, Any] = {}
            val_path = self.uargs.output_dir / f"validation_Epoch_{step}"
            mkdir(val_path)

            # ------------------------- BASENAME -------------------------
            name_keys = [
                "filename",
                "file_name",
                "image_path",
                "video_path",
                "path",
            ]
            base_name: str | None = None
            for key in name_keys:
                raw_val = batch.get(key)
                if not raw_val:
                    continue

                # Liste oder Tupel → erstes Element
                if isinstance(raw_val, (list, tuple)):
                    raw_val = raw_val[0] if raw_val else None
                if raw_val is None:
                    continue

                # Tensor → in Python‑Objekt gießen
                if torch.is_tensor(raw_val):
                    try:
                        raw_val = raw_val.item() if raw_val.ndim == 0 else raw_val.tolist()[0]
                    except Exception:  # noqa: BLE001
                        raw_val = str(raw_val)

                raw_str = str(raw_val)
                if raw_str:
                    base_name = Path(raw_str).stem
                    break

            if not base_name:
                base_name = f"processed{self.state.global_rank}-batch{i}"
            # -----------------------------------------------------------

            # ------------------------- SAVE ---------------------------
            (val_path / "").mkdir(parents=True, exist_ok=True)
            with open(val_path / f"{base_name}.txt", "w", encoding="utf-8") as f:
                f.write(prompt or "")

            img_res = val_res.get("image")
            if img_res is not None:
                img_path = val_path / f"{base_name}.png"
                img_res.save(img_path)
                artifacts["image"] = wandb.Image(str(img_path), caption=prompt)

            vid_res = val_res.get("video")
            if vid_res is not None:
                vid_path = val_path / f"{base_name}.mp4"
                export_to_video(vid_res, vid_path, fps=self.uargs.gen_fps)
                artifacts["video"] = wandb.Video(str(vid_path), caption=prompt)

            all_processes_artifacts.append(artifacts)

        # --------------------- Logging & Cleanup ---------------------
        if self.tracker is not None:
            gathered = gather_object(all_processes_artifacts)
            flat = [item for sublist in gathered for item in sublist]
            self.tracker.log({"validation": expand_list(flat)}, step=step)

        pipe.remove_all_hooks()
        del pipe
        self.move_components_to_device(
            dtype=self.state.weight_dtype,
            device=self.state.device,
            ignore_list=self.UNLOAD_LIST,
        )
        free_memory()
        dist.barrier()

        self.logger.info(
            "Memory after validation end: "
            + json.dumps(get_memory_statistics(self.logger), indent=4)
        )
        torch.cuda.reset_peak_memory_stats(self.state.device)
        torch.set_grad_enabled(True)
        self.components.transformer.train()

    # ------------------------------------------------------------------
    #                           EPOCH HOOK
    # ------------------------------------------------------------------
    @override
    def on_epoch_end(self, epoch: int, global_step: int) -> None:
        ckpt_path = self.maybe_save_checkpoint(global_step, must_save=True)
        if ckpt_path is not None:
            epoch_ckpt = self.uargs.output_dir / f"Checkpoint_Epoch_{epoch + 1}"
            if is_main_process():
                Path(ckpt_path).rename(epoch_ckpt)
        else:
            epoch_ckpt = None

        if self.uargs.do_validation and (epoch + 1) == self.uargs.train_epochs:
            free_memory()
            self.validate(epoch + 1, ckpt_path=epoch_ckpt)

    # ------------------------------------------------------------------
    #                       ABSTRACT / OVERRIDE METHODS
    # ------------------------------------------------------------------
    @override
    def load_components(self) -> DiffusionComponents:  # noqa: D401
        raise NotImplementedError

    @override
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def collate_fn(self, samples: list[dict[str, Any]]):  # noqa: D401, ANN001
        """Standard‑Collate‑Fn (Training & Validation).

        *  Für numerische / Tensor‑Werte verwenden wir `torch.utils.data.default_collate`.
        *  Nicht‑Tensor‑Werte (Strings, PIL‑Images, ``None`` usw.) werden zu
           Listen zusammengelegt.
        *  Das Feld ``filename`` (falls vorhanden) wird **immer** als Liste
           weitergereicht, damit der Validation‑Code es aus
           ``batch["filename"][0]`` auslesen kann.
        """
        from torch.utils.data._utils.collate import default_collate

        # Numerische / Tensor‑Spalten herausfiltern, Rest separat sammeln
        tensor_keys   = {}
        non_tensor    = {}
        for key in samples[0].keys():
            first_val = samples[0][key]
            if torch.is_tensor(first_val):
                tensor_keys[key] = [s[key] for s in samples]
            else:
                non_tensor[key]  = [s[key] for s in samples]

        # Standard‑Collate auf Tensor‑Spalten anwenden
        batch = {k: default_collate(v) for k, v in tensor_keys.items()}

        # Nicht‑Tensor‑Spalten anhängen (Listen beibehalten)
        batch.update(non_tensor)
        return batch

    def initialize_pipeline(self, ckpt_path: str | None = None) -> DiffusionPipeline:  # noqa: D401
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def get_negtive_prompt_embeds(self) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def validation_step(
        self, pipe: DiffusionPipeline, eval_data: dict[str, Any]
    ) -> dict[str, str | Image.Image | list[Image.Image]]:
        raise NotImplementedError

    # ---------------- PACKING ----------------
    def sample_to_length(self, sample: dict[str, Any]) -> int:
        raise NotImplementedError

    def collate_fn_packing(self, samples: list[dict[str, list[Any]]]) -> dict[str, Any]:
        """Collate‑Fn für Packing‑Sampler.

        Wie oben, nur dass hier jede Spalte bereits **Listen von Sub‑Samples**
        enthält. Wir müssen deshalb **eine Ebene tiefer** collaten.
        """
        from torch.utils.data._utils.collate import default_collate

        out: dict[str, Any] = {}
        for key in samples[0].keys():
            # Jedes samples[i][key] ist selbst eine Liste[SubSample]
            merged = sum([s[key] for s in samples], [])  # flat

            if torch.is_tensor(merged[0]):
                out[key] = default_collate(merged)  # Tensor collate
            else:
                out[key] = merged                    # Liste beibehalten
        return out
