import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

# ============================================================================
# 配置：每个训练阶段需要的 CSV 列 + 磁盘文件
# ============================================================================

STAGES = {
    "步骤1: SS VAE (SparseStructure)": {
        "csv_columns": {
            "required": ["sha256", "aesthetic_score", "voxelized"],
            "optional": [],
        },
        "files": {
            "voxels": {"dir": "voxels", "ext": ".ply", "desc": "64³ voxel grid"},
        },
        "filters_applied": [
            "voxelized == True",
            "aesthetic_score >= min_aesthetic_score (默认 4.5)",
        ],
    },
    "步骤2: SLat VAE (SparseFeat2Render)": {
        "csv_columns": {
            "required": [
                "sha256",
                "aesthetic_score",
                "feature_dinov2_vitl14_reg",
                "num_voxels",
            ],
            "optional": [],
        },
        "files": {
            "features/dinov2_vitl14_reg": {"dir": "features/dinov2_vitl14_reg", "ext": ".npz", "desc": "DINOv2 sparse features"},
            "renders": {"dir": "renders", "ext": "", "desc": "目录 (含 transforms.json)"},
        },
        "filters_applied": [
            "feature_dinov2_vitl14_reg == True",
            "aesthetic_score >= min_aesthetic_score",
            "num_voxels <= max_num_voxels (默认 32768)",
        ],
    },
    "步骤5: SS Flow (TextConditionedSparseStructureLatent)": {
        "csv_columns": {
            "required": [
                "sha256",
                "aesthetic_score",
                "ss_latent_ss_enc_conv3d_16l8_fp16",
                "captions",
            ],
            "optional": [],
        },
        "files": {
            "ss_latents/ss_enc_conv3d_16l8_fp16": {"dir": "ss_latents/ss_enc_conv3d_16l8_fp16", "ext": ".npz", "desc": "SS latent"},
        },
        "filters_applied": [
            "ss_latent_ss_enc_conv3d_16l8_fp16 == True",
            "aesthetic_score >= min_aesthetic_score",
            "captions 不为空",
        ],
    },
    "步骤6: SLat Flow (TextConditionedSLat)": {
        "csv_columns": {
            "required": [
                "sha256",
                "aesthetic_score",
                "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16",
                "captions",
                "num_voxels",
            ],
            "optional": [],
        },
        "files": {
            "latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16": {"dir": "latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16", "ext": ".npz", "desc": "SLat"},
        },
        "filters_applied": [
            "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16 == True",
            "aesthetic_score >= min_aesthetic_score",
            "num_voxels <= max_num_voxels",
            "captions 不为空",
        ],
    },
}


def load_metadata(data_dir: str) -> pd.DataFrame:
    """Load metadata.csv"""
    csv_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(csv_path):
        print(f"\n❌ 致命错误: 找不到 {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"\n📄 加载 metadata.csv: {len(df)} 行, {len(df.columns)} 列")
    return df


def check_sha256(df: pd.DataFrame) -> list:
    """验证 sha256 列"""
    errors = []
    if "sha256" not in df.columns:
        errors.append("缺少 sha256 列 (必须)")
        return errors

    null_count = df["sha256"].isna().sum()
    dup_count = df["sha256"].duplicated().sum()
    if null_count > 0:
        errors.append(f"sha256 有 {null_count} 个空值")
    if dup_count > 0:
        dup_vals = df[df["sha256"].duplicated()]["sha256"].values
        errors.append(f"sha256 有 {dup_count} 个重复值: {dup_vals[:5]}...")

    if not errors:
        print(f"  ✅ sha256: {len(df)} 个唯一值，无空值，无重复")
    return errors


def check_columns(df: pd.DataFrame, stage_name: str, spec: dict) -> list:
    """验证某阶段所需的列"""
    errors = []
    for col in spec["required"]:
        if col not in df.columns:
            errors.append(f"缺少必需列: '{col}'")
        else:
            null_count = df[col].isna().sum()
            if null_count > 0:
                errors.append(f"'{col}' 有 {null_count} 个空值")
            # 类型/值域检查
            if col == "aesthetic_score":
                neg = (df[col] < 0).sum()
                if neg > 0:
                    errors.append(f"'{col}' 有 {neg} 个负数")
            if col == "num_voxels":
                neg = (df[col] <= 0).sum()
                if neg > 0:
                    errors.append(f"'{col}' 有 {neg} 个 ≤0 的值")
            if col == "captions":
                bad = 0
                for i, v in df[col].dropna().items():
                    try:
                        parsed = json.loads(v)
                        if not isinstance(parsed, list) or len(parsed) == 0:
                            bad += 1
                    except (json.JSONDecodeError, TypeError):
                        bad += 1
                if bad > 0:
                    errors.append(f"'{col}' 有 {bad} 行 JSON 格式不正确或为空数组")

    return errors


def check_files(df: pd.DataFrame, data_dir: str, stage_name: str, spec: dict) -> list:
    """验证磁盘文件存在性"""
    errors = []
    sha256_vals = df["sha256"].dropna().values

    for label, finfo in spec.get("files", {}).items():
        dir_path = os.path.join(data_dir, finfo["dir"])
        if not os.path.exists(dir_path):
            errors.append(f"目录不存在: {dir_path}")
            continue

        sample_ids = list(sha256_vals[:5]) + list(np.random.choice(sha256_vals, min(5, len(sha256_vals)), replace=False))
        missing = []
        for sid in set(sample_ids):
            if finfo["ext"] == "":
                target = os.path.join(dir_path, str(sid))
                if not os.path.isdir(target):
                    missing.append(sid)
                elif not os.path.exists(os.path.join(target, "transforms.json")):
                    missing.append(f"{sid}/transforms.json")
            else:
                target = os.path.join(dir_path, f"{sid}{finfo['ext']}")
                if not os.path.exists(target):
                    missing.append(sid)

        if missing:
            errors.append(f"{label}: {len(missing)}/{len(set(sample_ids))} 采样缺失, 例: {missing[:3]}")
        else:
            print(f"  ✅ {label}: 采样 {len(set(sample_ids))} 个全部存在")

    return errors


def check_filter_counts(df: pd.DataFrame, stage_name: str, spec: dict) -> None:
    """模拟 TRELLIS filter_metadata"""
    working = df.copy()

    for col in spec["csv_columns"]["required"]:
        if col not in working.columns:
            print(f"  ⚠️  跳过过滤 (缺列 '{col}')")
            return
        if col in ("sha256",):
            continue
        if col == "aesthetic_score":
            before = len(working)
            working = working[working[col] >= 4.5]
            print(f"  ├─ aesthetic_score >= 4.5: {before} → {len(working)}")
        elif col == "num_voxels":
            before = len(working)
            working = working[working[col] <= 32768]
            print(f"  ├─ num_voxels <= 32768: {before} → {len(working)}")
        elif col == "captions":
            before = len(working)
            working = working[working[col].notna()]
            print(f"  ├─ captions 非空: {before} → {len(working)}")
        elif col.endswith("_reg") or col.endswith("_fp16"):
            before = len(working)
            working = working[working[col] == True]
            print(f"  ├─ {col} == True: {before} → {len(working)}")
        else:
            before = len(working)
            working = working[working[col] == True]
            print(f"  ├─ {col} == True: {before} → {len(working)}")

    print(f"  └─ ✅ 最终可用样本: {len(working)}")


def check_train_test_split(data_dir: str, df: pd.DataFrame) -> None:
    """检查 train.json / test.json 划分"""
    for split_name, split_file in [("训练集", "train.json"), ("测试集", "test.json")]:
        path = os.path.join(data_dir, split_file)
        if os.path.exists(path):
            try:
                ids = json.load(open(path))
                in_csv = sum(1 for i in ids if i in df["file_identifier"].values
                             or i in df["ID"].astype(str).values
                             or i in df["sha256"].values)
                print(f"  ✅ {split_name}: {split_file} 有 {len(ids)} 个 ID, {in_csv}/{len(ids)} 在 metadata 中")
            except Exception as e:
                print(f"  ⚠️  {split_name}: 读取 {split_file} 失败: {e}")
        else:
            print(f"  ⚠️  {split_name}: 找不到 {split_file} (非阻塞)")


def main():
    parser = argparse.ArgumentParser(description="验证 metadata.csv 完整性")
    parser.add_argument("--data_dir", type=str, default="./database/",
                        help="数据目录路径 (含 metadata.csv)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print(f"🔍 验证目录: {data_dir}")
    print("=" * 60)

    df = load_metadata(data_dir)

    print("\n" + "=" * 60)
    print("🔑 主键检查 (sha256)")
    print("-" * 40)
    sha_errors = check_sha256(df)

    total_errors = 0
    for stage_name, spec in STAGES.items():
        print("\n" + "=" * 60)
        print(f"📋 {stage_name}")
        print("-" * 40)

        col_errors = check_columns(df, stage_name, spec["csv_columns"])
        for e in col_errors:
            print(f"  ❌ {e}")
        if not col_errors:
            print(f"  ✅ 必需列全部存在且格式正确")

        file_errors = check_files(df, data_dir, stage_name, spec)

        print(f"\n  📊 过滤链模拟:")
        check_filter_counts(df, stage_name, spec)

        stage_errors = len(col_errors) + len(file_errors)
        total_errors += stage_errors
        if stage_errors == 0:
            print(f"\n  🎉 全部通过!")
        else:
            print(f"\n  ⚠️  {stage_errors} 个问题")

    print("\n" + "=" * 60)
    print("📂 训练/测试划分")
    print("-" * 40)
    check_train_test_split(data_dir, df)

    print("\n" + "=" * 60)
    print("📊 数据统计摘要")
    print("-" * 40)
    print(f"  总行数:              {len(df)}")
    print(f"  aesthetic_score 范围: {df['aesthetic_score'].min():.1f} ~ {df['aesthetic_score'].max():.1f}")
    if "num_voxels" in df.columns:
        nv = df["num_voxels"].dropna()
        print(f"  num_voxels 范围:      {int(nv.min())} ~ {int(nv.max())} (均值 {nv.mean():.0f})")
    if "captions" in df.columns:
        has_cap = df["captions"].notna().sum()
        print(f"  有 captions:          {has_cap}/{len(df)}")
    if "voxelized" in df.columns:
        has_vox = (df["voxelized"] == True).sum()
        print(f"  已体素化:             {has_vox}/{len(df)}")
    if "rendered" in df.columns:
        has_ren = (df["rendered"] == True).sum()
        print(f"  已渲染:               {has_ren}/{len(df)}")
    if "feature_dinov2_vitl14_reg" in df.columns:
        has_feat = (df["feature_dinov2_vitl14_reg"] == True).sum()
        print(f"  有 DINO 特征:         {has_feat}/{len(df)}")
    if "ss_latent_ss_enc_conv3d_16l8_fp16" in df.columns:
        has_ss = (df["ss_latent_ss_enc_conv3d_16l8_fp16"] == True).sum()
        print(f"  有 SS latent:         {has_ss}/{len(df)}")
    if "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16" in df.columns:
        has_sl = (df["latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"] == True).sum()
        print(f"  有 SLat:              {has_sl}/{len(df)}")

    print("\n" + "=" * 60)
    if total_errors == 0 and len(sha_errors) == 0:
        print("✅ 全部检查通过！metadata.csv 可以直接用于训练。")
    else:
        print(f"⚠️  发现 {total_errors + len(sha_errors)} 个问题需要修复。")
        if sha_errors:
            print("\n  sha256 问题:")
            for e in sha_errors:
                print(f"    - {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
