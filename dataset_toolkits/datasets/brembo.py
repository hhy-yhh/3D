import os
import pandas as pd
from easydict import EasyDict as edict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


#汽车卡钳
# 该文件包含处理 Brembo 数据集的实用程序函数，包括加载元数据、并行处理实例以及从元数据中提取特定字段的函数。
def add_args(parser):
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--metadata', type=str, default='metadata.csv', help='Metadata file name')

# 加载元数据--Load metadata
def load_meta(args):
    metadata_path = os.path.join(args.data_dir, args.metadata)
    df = pd.read_csv(metadata_path)
    
    if 'rendered' in df.columns:
        df['rendered'] = df['rendered'].astype(str).str.lower()
        df = df[df['rendered'] == 'true']
    
    if 'voxelized' in df.columns:
        df['voxelized'] = df['voxelized'].astype(str).str.lower()
        df = df[df['voxelized'] != 'true']
    
    meta = []
    for _, row in df.iterrows():
        meta.append(edict({
            'sha256': row['sha256'],
            'file_identifier': row['file_identifier'],
            'file_path': row['file_path'],
            'local_path': row['file_path'],
            'aesthetic_score': row.get('aesthetic_score', 5.0),
        }))
    return meta

# 并行处理实例--Parallel processing of instances
def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    metadata = metadata.to_dict('records')
    records = []
    max_workers = max_workers or os.cpu_count()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
        tqdm(total=len(metadata), desc=desc) as pbar:
        def worker(metadatum):
            try:
                sha256 = metadatum['sha256']
                file_path = metadatum['file_path']
                # 不传入 output_dir，让 _voxelize 自己从全局获取
                record = func(file_path, sha256)
                if record is not None:
                    records.append(record)
            except Exception as e:
                print(f'Error processing {metadatum.get("sha256", "unknown")}: {e}')
            finally:
                pbar.update(1)
        
        futures = [executor.submit(worker, metadatum) for metadatum in metadata]
        for future in futures:
            future.result()
    
    return pd.DataFrame.from_records(records)

def get_sha256(meta):
    return meta.sha256

def get_file_path(meta):
    return meta.file_path

def get_file_identifier(meta):
    return meta.file_identifier

def get_aesthetic_score(meta):
    return meta.aesthetic_score
