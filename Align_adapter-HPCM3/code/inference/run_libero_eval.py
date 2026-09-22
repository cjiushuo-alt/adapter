"""
run_libero_eval.py (Align_adapter-HPCM2)

仿照 Align_adapter/inference/run_libero_eval.py：接口与配置一致，使用 HPCM + Adapter 推理模型。
"""

import json
import hashlib
import logging
import os
import random
import subprocess
import sys
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# 仓库与组件路径
hpcm3_root = Path(__file__).resolve().parents[2]
repo_root = Path(__file__).resolve().parents[3]
libero_path = repo_root / "LIBERO"
if str(libero_path) not in sys.path:
    sys.path.insert(0, str(libero_path))

import yaml
libero_config_path = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
config_file = os.path.join(libero_config_path, "config.yaml")
if not os.path.exists(config_file):
    os.makedirs(libero_config_path, exist_ok=True)
    benchmark_root_path = str(libero_path / "libero" / "libero")
    default_path_dict = {
        "benchmark_root": benchmark_root_path,
        "bddl_files": os.path.join(benchmark_root_path, "bddl_files"),
        "init_states": os.path.join(benchmark_root_path, "init_files"),
        "datasets": os.path.join(benchmark_root_path, "..", "datasets"),
        "assets": os.path.join(benchmark_root_path, "assets"),
    }
    with open(config_file, "w") as f:
        yaml.dump(default_path_dict, f)

import draccus
import numpy as np
import tqdm
import torch
from libero.libero import benchmark

try:
    import wandb
except ImportError:
    wandb = None

project_root = hpcm3_root
for path in (repo_root, project_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

hpcm2_root = Path(__file__).resolve().parent.parent
if str(hpcm2_root) not in sys.path:
    sys.path.insert(0, str(hpcm2_root))
from inference.model.model import get_custom_vla_model

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------- 配置 (与 Align_adapter 结构一致) ----------
@dataclass
class HPCMConfig:
    hpcm_root: str = ""
    checkpoint: str = ""
    model_name: str = "HPCM_Base"
    scale_table_levels: int = 60


@dataclass
class AdapterConfig:
    """与 train_config 一致，用于构建 Adapter；未在 YAML 中配置的项由代码默认值兜底。"""
    checkpoint: str = ""
    hpcm_input_dim: int = 320
    pre_neck_output_dim: int = 320
    pre_neck_spatial_size: int = 16
    pre_neck_num_res_blocks: int = 2
    pre_neck_dropout: float = 0.0
    siglip_dim: int = 1152
    dino_dim: int = 1024
    neck_num_heads: int = 8
    neck_ffn_hidden_dim: Optional[int] = None
    neck_dropout: float = 0.0
    neck_activation: str = "gelu"


@dataclass
class VisionBackboneConfig:
    max_layer: int = 2


@dataclass
class ModelConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    # Keep False for exact parity with the successful historical Spatial run.
    # True enables the raw 256px HPCM-input diagnostic path.
    hpcm_raw_input_bypass: bool = False
    num_open_loop_steps: int = 8
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    save_version: str = "vla-adapter-hpcm2"
    use_pro_version: bool = True
    hpcm: HPCMConfig = None
    adapter: AdapterConfig = None
    vision_backbone: VisionBackboneConfig = None
    vla_path: str = "prismatic"
    vla_id: str = "prismatic"
    processor_name: Optional[str] = None

    def __post_init__(self):
        if self.hpcm is None:
            self.hpcm = HPCMConfig()
        if self.adapter is None:
            self.adapter = AdapterConfig()
        if self.vision_backbone is None:
            self.vision_backbone = VisionBackboneConfig()


@dataclass
class LiberoConfig:
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256


@dataclass
class LoggingConfig:
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    episode_results_jsonl: Optional[str] = None
    summary_json: Optional[str] = None
    save_rollout_videos: bool = False


@dataclass
class AuditConfig:
    """可选的数据重合审计；默认关闭，不影响正常评测。"""
    enabled: bool = False
    output_jsonl: Optional[str] = None
    compare_train_frames: bool = False
    train_dataset_dir: str = "Align_adapter/dataset"
    train_ratio: float = 0.9
    split_seed: int = 42
    train_hash_cache: Optional[str] = None
    split_manifest_path: Optional[str] = None


@dataclass
class GuardConfig:
    """Stop a formal run early when it clearly needs human review."""
    enabled: bool = False
    min_episodes: int = 20
    min_success_rate: float = 0.5


@dataclass
class InferenceConfig:
    model: ModelConfig
    libero: LiberoConfig
    logging: LoggingConfig
    audit: AuditConfig = None
    guard: GuardConfig = None
    seed: int = 7
    phase: str = "Inference"
    startup_check_only: bool = False

    def __post_init__(self):
        if self.audit is None:
            self.audit = AuditConfig()
        if self.guard is None:
            self.guard = GuardConfig()

    @property
    def model_family(self):
        return self.model.model_family

    @property
    def pretrained_checkpoint(self):
        return self.model.pretrained_checkpoint

    @property
    def task_suite_name(self):
        return self.libero.task_suite_name

    @property
    def run_id_note(self):
        return self.logging.run_id_note

    @property
    def local_log_dir(self):
        return self.logging.local_log_dir

    @property
    def use_wandb(self):
        return self.logging.use_wandb

    @property
    def wandb_entity(self):
        return self.logging.wandb_entity

    @property
    def wandb_project(self):
        return self.logging.wandb_project

    @property
    def num_images_in_input(self):
        return self.model.num_images_in_input

    @property
    def use_proprio(self):
        return self.model.use_proprio

    @property
    def unnorm_key(self):
        return getattr(self.model, "unnorm_key", "")

    @property
    def num_open_loop_steps(self):
        return self.model.num_open_loop_steps

    @property
    def use_l1_regression(self):
        return self.model.use_l1_regression

    @property
    def use_pro_version(self):
        return self.model.use_pro_version

    @property
    def save_version(self):
        return self.model.save_version

    @property
    def center_crop(self):
        return self.model.center_crop

    @property
    def hpcm_raw_input_bypass(self):
        return self.model.hpcm_raw_input_bypass

def _resolve_vla_checkpoint(vla_path: str, pretrained_checkpoint: str) -> str:
    """
    解析 VLA checkpoint 路径。返回：
    - prismatic 格式：checkpoints/step-*.pt 的绝对路径
    - HF 格式：含 config.json 和 model.safetensors 的目录绝对路径（供 model 中 HF 加载器使用）
    """
    p = Path(vla_path)
    if not p.is_absolute():
        repo_candidate = repo_root / p
        p = repo_candidate if repo_candidate.exists() else project_root / p
    if p.is_file() and p.suffix == ".pt" and p.parent.name == "checkpoints":
        return str(p)
    if p.is_dir():
        ckpt_dir = p / "checkpoints"
        if ckpt_dir.exists():
            pts = list(ckpt_dir.glob("step-*.pt")) or list(ckpt_dir.glob("*.pt"))
            if pts:
                latest = max(pts, key=lambda x: x.stat().st_mtime)
                return str(latest)
        if (p / "config.json").exists() and ((p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()):
            return str(p)
    return str(p) if p.exists() else vla_path


def validate_config(cfg: InferenceConfig) -> None:
    assert cfg.model.pretrained_checkpoint, "pretrained_checkpoint must be set."
    assert not (cfg.model.load_in_8bit and cfg.model.load_in_4bit), "Cannot use both 8-bit and 4-bit."
    assert cfg.libero.task_suite_name in [s.value for s in TaskSuite], (
        f"Invalid task_suite_name: {cfg.libero.task_suite_name}"
    )
    assert cfg.libero.num_trials_per_task == 50, "Formal LIBERO-4 evaluation requires 50 episodes per task."
    for label, raw_path in (
        ("HPCM checkpoint", cfg.model.hpcm.checkpoint),
        ("HPCM3 Adapter checkpoint", cfg.model.adapter.checkpoint),
    ):
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        assert path.is_file(), f"{label} not found: {path}"


def get_custom_model(cfg: InferenceConfig, device: torch.device):
    """构建 HPCM3：get_vla + VisionBackboneWrapper(CombinedVisionModel)，Adapter 使用 ResNet PreNeck。"""
    hpcm = cfg.model.hpcm
    adapter = cfg.model.adapter
    vb = cfg.model.vision_backbone
    vla_path = cfg.model.vla_path or cfg.model.pretrained_checkpoint
    vla_path = _resolve_vla_checkpoint(vla_path, cfg.model.pretrained_checkpoint)
    if not Path(vla_path).is_file():
        vla_path = _resolve_vla_checkpoint(cfg.model.pretrained_checkpoint, cfg.model.pretrained_checkpoint)

    hpcm_root = hpcm.hpcm_root or str(repo_root / "HPCM")
    hpcm_checkpoint = hpcm.checkpoint
    if hpcm_checkpoint and not Path(hpcm_checkpoint).is_absolute():
        hpcm_checkpoint = str(Path(hpcm_root) / hpcm_checkpoint)

    adapter_path = adapter.checkpoint
    if adapter_path and not Path(adapter_path).is_absolute():
        adapter_path = str(repo_root / adapter_path)
    adapter_config = {k: v for k, v in asdict(adapter).items() if k != "checkpoint"}

    model = get_custom_vla_model(
        openvla_path=vla_path,
        vision_backbone_id="dinosiglip-vit-so-224px",
        vision_backbone_checkpoint=None,
        vision_backbone_max_layer=vb.max_layer,
        hpcm_root=hpcm_root,
        hpcm_checkpoint=hpcm_checkpoint or None,
        adapter_weights_path=adapter_path or None,
        adapter_config=adapter_config,
        load_in_8bit=cfg.model.load_in_8bit,
        load_in_4bit=cfg.model.load_in_4bit,
        use_film=cfg.model.use_film,
        num_images_in_input=cfg.model.num_images_in_input,
        device=device,
    )
    return model


def check_unnorm_key(cfg: InferenceConfig, model) -> None:
    # A fixed key allows a single frozen VLA checkpoint to be evaluated across suites
    # without modifying or fabricating checkpoint statistics.
    unnorm_key = cfg.model.unnorm_key or cfg.libero.task_suite_name
    norm_stats = getattr(model, "norm_stats", {})
    if unnorm_key not in norm_stats and f"{unnorm_key}_no_noops" in norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in norm_stats, f"unnorm_key {unnorm_key} not in model norm_stats: {list(norm_stats.keys())}"
    cfg.model.unnorm_key = unnorm_key


def initialize_model(cfg: InferenceConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_custom_model(cfg, device)
    model = model.to(device)
    model.eval()
    if hasattr(model, "set_version"):
        model.set_version(cfg.model.save_version)
    check_unnorm_key(cfg, model)
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.model.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if cfg.model.use_l1_regression else None
    noisy_action_projector = None
    return model, action_head, proprio_projector, noisy_action_projector, processor


def setup_logging(cfg: InferenceConfig):
    run_id = f"EVAL-{cfg.libero.task_suite_name}-{cfg.model.model_family}-{DATE_TIME}"
    if cfg.logging.run_id_note:
        run_id += f"--{cfg.logging.run_id_note}"
    os.makedirs(cfg.logging.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.logging.local_log_dir, "run.log")
    log_file = open(local_log_filepath, "x", encoding="utf-8")
    logger.info("Logging to %s", local_log_filepath)
    if cfg.logging.use_wandb and wandb is not None:
        wandb.init(
            entity=cfg.logging.wandb_entity,
            project=cfg.logging.wandb_project,
            name=run_id,
        )
    return log_file, local_log_filepath, run_id


def log_message(msg: str, log_file=None):
    logger.info(msg)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def _sha256_rgb(image: np.ndarray) -> str:
    """对解码后的 RGB 像素字节哈希，避免 PNG/JPEG 编码差异影响 exact-frame 比较。"""
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    """对 initial-state 的 dtype、shape 和原始数值字节做稳定哈希。"""
    arr = np.ascontiguousarray(np.asarray(array))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _git_metadata() -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def _checkpoint_metadata(cfg: InferenceConfig) -> Dict[str, Any]:
    adapter_path = _resolve_repo_path(cfg.model.adapter.checkpoint).resolve()
    hpcm_path = _resolve_repo_path(cfg.model.hpcm.checkpoint).resolve()
    vla_dir = _resolve_repo_path(cfg.model.pretrained_checkpoint).resolve()
    vla_files = {}
    for name in ("model.safetensors", "action_head--checkpoint.pt", "proprio_projector--checkpoint.pt"):
        path = vla_dir / name
        if path.is_file():
            vla_files[name] = {"path": str(path), "sha256": _sha256_file(path)}
    return {
        "hpcm3_adapter": {"path": str(adapter_path), "sha256": _sha256_file(adapter_path)},
        "hpcm": {"path": str(hpcm_path), "sha256": _sha256_file(hpcm_path)},
        "vla": {"path": str(vla_dir), "files": vla_files},
    }


def build_training_frame_hashes(cfg: InferenceConfig) -> Optional[set]:
    """按需扫描训练候选语料的 PNG；代价较高，因此必须显式开启。"""
    if not cfg.audit.enabled or not cfg.audit.compare_train_frames:
        return None
    from PIL import Image

    if cfg.audit.train_hash_cache:
        cache_path = _resolve_repo_path(cfg.audit.train_hash_cache)
        if cache_path.is_file():
            with cache_path.open("r", encoding="ascii") as handle:
                hashes = {line.strip() for line in handle if line.strip()}
            if not hashes or any(len(value) != 64 for value in hashes):
                raise ValueError(f"Invalid training-frame hash cache: {cache_path}")
            log_message(f"Audit: loaded {len(hashes)} unique training-frame hashes from {cache_path}")
            return hashes

    root = Path(cfg.audit.train_dataset_dir)
    if not root.is_absolute():
        root = project_root / root
        if not root.exists():
            # project_root 是 Align_adapter-HPCM3；默认训练图像实际位于 repo/Align_adapter/dataset。
            repo_candidate = repo_root / cfg.audit.train_dataset_dir
            if repo_candidate.exists():
                root = repo_candidate
    if cfg.audit.split_manifest_path:
        manifest_path = _resolve_repo_path(cfg.audit.split_manifest_path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("split_unit") != "trajectory":
            raise ValueError(f"Not a trajectory split manifest: {manifest_path}")
        paths = []
        for episode in manifest.get("train_episodes", []):
            paths.extend(Path(episode).glob("step_*_image.png"))
        paths = sorted(set(paths))
    else:
        paths = sorted(root.glob("libero_*_no_noops/episode_*/step_*_image.png"))
    if not paths:
        raise FileNotFoundError(f"Audit training frames not found under: {root}")
    if not cfg.audit.split_manifest_path:
        # 仅用于复现旧的 frame-level split。
        indices = list(range(len(paths)))
        random.Random(cfg.audit.split_seed).shuffle(indices)
        split_idx = int(len(indices) * cfg.audit.train_ratio)
        paths = [paths[i] for i in indices[:split_idx]]
    log_message(f"Audit: hashing {len(paths)} decoded training RGB frames under {root}")
    hashes = set()
    for idx, path in enumerate(paths, 1):
        with Image.open(path) as image:
            hashes.add(_sha256_rgb(np.asarray(image.convert("RGB"))))
        if idx % 50000 == 0:
            log_message(f"Audit: hashed {idx}/{len(paths)} training frames")
    log_message(f"Audit: built {len(hashes)} unique training-frame hashes")
    return hashes


def setup_audit(cfg: InferenceConfig, run_id: str):
    if not cfg.audit.enabled:
        return None
    output = cfg.audit.output_jsonl
    if output:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    else:
        output_path = Path(cfg.logging.local_log_dir) / f"{run_id}--frame-audit.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Frame audit enabled: %s", output_path)
    return output_path.open("x", encoding="utf-8")


def setup_episode_results(cfg: InferenceConfig):
    output = cfg.logging.episode_results_jsonl
    output_path = _resolve_repo_path(output) if output else Path(cfg.logging.local_log_dir) / "episode_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("x", encoding="utf-8")


def write_audit_record(audit_file, record: Dict[str, Any]) -> None:
    if audit_file is None:
        return
    audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    audit_file.flush()


def load_initial_states(cfg: InferenceConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.libero.initial_states_path != "DEFAULT":
        with open(cfg.libero.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.libero.initial_states_path}", log_file)
        return initial_states, all_initial_states
    log_message("Using default initial states", log_file)
    return initial_states, None


def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_resized,
        # Preserve the decoded environment frames for HPCM.  The normal fields
        # remain unchanged because the OpenVLA processor is still used to build
        # the remaining model inputs.
        "_hpcm_raw_images": [img, wrist_img],
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img, wrist_img


def process_action(action: np.ndarray, model_family: str) -> np.ndarray:
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_episode(
    cfg: InferenceConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    audit_file=None,
    audit_context: Optional[Dict[str, Any]] = None,
    training_frame_hashes: Optional[set] = None,
):
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.model.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        logger.warning(
            "num_open_loop_steps (%s) != NUM_ACTIONS_CHUNK (%s)",
            cfg.model.num_open_loop_steps,
            NUM_ACTIONS_CHUNK,
        )
    action_queue = deque(maxlen=cfg.model.num_open_loop_steps)
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS.get(cfg.libero.task_suite_name, 300)
    success = False
    exact_train_frame_matches = 0

    try:
        while t < max_steps + cfg.libero.num_steps_wait:
            if t < cfg.libero.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model.model_family))
                t += 1
                continue

            observation, img, wrist_img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if audit_file is not None:
                main_hash = _sha256_rgb(img)
                wrist_hash = _sha256_rgb(wrist_img)
                main_match = main_hash in training_frame_hashes if training_frame_hashes is not None else None
                wrist_match = wrist_hash in training_frame_hashes if training_frame_hashes is not None else None
                exact_train_frame_matches += int(main_match is True) + int(wrist_match is True)
                write_audit_record(
                    audit_file,
                    {
                        **(audit_context or {}),
                        "rollout_step": t - cfg.libero.num_steps_wait,
                        "initial_state_sha256": _sha256_array(initial_state) if initial_state is not None else None,
                        "main_rgb_sha256": main_hash,
                        "wrist_rgb_sha256": wrist_hash,
                        "main_rgb_shape": list(img.shape),
                        "wrist_rgb_shape": list(wrist_img.shape),
                        "main_exact_train_match": main_match,
                        "wrist_exact_train_match": wrist_match,
                    },
                )

            if len(action_queue) == 0:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.model.use_film,
                    use_minivlm=cfg.model.use_minivlm,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            # 与 Align_adapter 一致：保证为 numpy，避免 tensor 导致 normalize_gripper_action 报错
            action = np.asarray(action, dtype=np.float64) if not isinstance(action, np.ndarray) else action
            action = process_action(action, cfg.model.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        import traceback
        traceback.print_exc()

    return success, replay_images, exact_train_frame_matches


def run_task(
    cfg: InferenceConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    save_version=None,
    audit_file=None,
    training_frame_hashes: Optional[set] = None,
    episode_results_file=None,
):
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model.model_family, resolution=cfg.libero.env_img_res)

    task_episodes, task_successes = 0, 0
    task_exact_matches = 0
    review_required = False
    for episode_idx in tqdm.tqdm(range(cfg.libero.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.libero.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            key = task_description.replace(" ", "_")
            ep_key = f"demo_{episode_idx}"
            if not all_initial_states[key][ep_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} (failed expert demo)", log_file)
                continue
            initial_state = np.array(all_initial_states[key][ep_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        success, replay_images, episode_exact_matches = run_episode(
            cfg, env, task_description, model, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            initial_state, log_file,
            audit_file=audit_file,
            audit_context={
                "task_suite": cfg.libero.task_suite_name,
                "task_id": task_id,
                "task_description": task_description,
                "episode_index": episode_idx,
                "initial_state_source": cfg.libero.initial_states_path,
            },
            training_frame_hashes=training_frame_hashes,
        )
        task_episodes += 1
        total_episodes += 1
        task_exact_matches += episode_exact_matches
        if success:
            task_successes += 1
            total_successes += 1

        if cfg.logging.save_rollout_videos:
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                save_version=save_version,
                root_dir=cfg.logging.local_log_dir,
            )
        write_audit_record(
            episode_results_file,
            {
                "suite": cfg.libero.task_suite_name,
                "task_id": task_id,
                "task_description": task_description,
                "episode_index": episode_idx,
                "initial_state_source": cfg.libero.initial_states_path,
                "initial_state_sha256": _sha256_array(initial_state),
                "success": bool(success),
                "exact_train_frame_matches": episode_exact_matches,
            },
        )
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({100.0 * total_successes / total_episodes:.1f}%)", log_file)

        if (
            cfg.guard.enabled
            and total_episodes >= cfg.guard.min_episodes
            and total_successes / total_episodes < cfg.guard.min_success_rate
        ):
            review_required = True
            log_message(
                "REVIEW REQUIRED: success rate "
                f"{total_successes / total_episodes:.4f} after {total_episodes} episodes is below "
                f"guard threshold {cfg.guard.min_success_rate:.4f}",
                log_file,
            )
            break

    task_sr = float(task_successes) / task_episodes if task_episodes > 0 else 0
    total_sr = float(total_successes) / total_episodes if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_sr}", log_file)
    log_message(f"Current total success rate: {total_sr}", log_file)
    env.close()
    del env

    if cfg.logging.use_wandb and wandb is not None:
        wandb.log(
            {f"success_rate/{task_description}": task_sr, f"num_episodes/{task_description}": task_episodes}
        )
    return total_episodes, total_successes, {
        "task_id": task_id,
        "task_description": task_description,
        "episodes": task_episodes,
        "success_episodes": task_successes,
        "success_rate": task_sr,
        "exact_train_frame_matches": task_exact_matches,
    }, review_required


@draccus.wrap()
def eval_libero(cfg: InferenceConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    if cfg.startup_check_only:
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        task_suite = benchmark.get_benchmark_dict()[cfg.libero.task_suite_name]()
        assert task_suite.n_tasks == 10, f"Expected 10 tasks, found {task_suite.n_tasks}"
        initial_states = task_suite.get_task_init_states(0)
        assert len(initial_states) >= cfg.libero.num_trials_per_task
        env, description = get_libero_env(
            task_suite.get_task(0), cfg.model.model_family, resolution=cfg.libero.env_img_res
        )
        obs = env.set_init_state(initial_states[0])
        resize_size = get_image_resize_size(cfg)
        prepare_observation(obs, resize_size)
        env.close()
        print(
            f"STARTUP_CHECK_OK suite={cfg.libero.task_suite_name} tasks={task_suite.n_tasks} "
            f"states_task0={len(initial_states)} task0={description!r} unnorm_key={cfg.model.unnorm_key}"
        )
        return 0.0

    log_file, local_log_filepath, run_id = setup_logging(cfg)
    episode_results_file = setup_episode_results(cfg)
    log_message(f"Git metadata: {_git_metadata()}", log_file)
    log_message(f"Fixed seed: {cfg.seed}", log_file)
    log_message(f"VLA action unnorm key: {cfg.model.unnorm_key}", log_file)
    checkpoint_metadata = _checkpoint_metadata(cfg)
    log_message(f"Checkpoint metadata: {json.dumps(checkpoint_metadata, ensure_ascii=False)}", log_file)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    training_frame_hashes = build_training_frame_hashes(cfg)
    audit_file = setup_audit(cfg, run_id)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.libero.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.libero.task_suite_name}", log_file)
    total_episodes, total_successes = 0, 0
    task_results = []
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes, task_result, review_required = run_task(
            cfg, task_suite, task_id, model, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            total_episodes, total_successes, log_file, cfg.model.save_version,
            audit_file, training_frame_hashes, episode_results_file,
        )
        task_results.append(task_result)
        if review_required:
            review_path = Path(cfg.logging.local_log_dir) / "review_required.json"
            review = {
                "suite": cfg.libero.task_suite_name,
                "reason": "success_rate_below_guard_threshold",
                "episodes": total_episodes,
                "success_episodes": total_successes,
                "success_rate": total_successes / total_episodes,
                "guard": asdict(cfg.guard),
                "last_task": task_result,
            }
            with review_path.open("x", encoding="utf-8") as handle:
                json.dump(review, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            if audit_file:
                audit_file.close()
            episode_results_file.close()
            log_file.close()
            raise RuntimeError(f"Formal evaluation halted for review; see {review_path}")

    final_sr = float(total_successes) / total_episodes if total_episodes > 0 else 0
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_sr:.4f} ({100 * final_sr:.1f}%)", log_file)

    summary = {
        "suite": cfg.libero.task_suite_name,
        "checkpoint_path": checkpoint_metadata["hpcm3_adapter"]["path"],
        "checkpoint_sha256": checkpoint_metadata["hpcm3_adapter"]["sha256"],
        "checkpoints": checkpoint_metadata,
        "git": _git_metadata(),
        "seed": cfg.seed,
        "action_unnorm_key": cfg.model.unnorm_key,
        "task_count": num_tasks,
        "total_episodes": total_episodes,
        "success_episodes": total_successes,
        "success_rate": final_sr,
        "per_task": task_results,
        "exact_train_frame_matches": sum(item["exact_train_frame_matches"] for item in task_results),
    }
    summary_path = (
        _resolve_repo_path(cfg.logging.summary_json)
        if cfg.logging.summary_json
        else Path(cfg.logging.local_log_dir) / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if cfg.logging.use_wandb and wandb is not None:
        wandb.log({"success_rate/total": final_sr, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)
    if log_file:
        log_file.close()
    if audit_file:
        audit_file.close()
    episode_results_file.close()
    return final_sr


if __name__ == "__main__":
    eval_libero()
