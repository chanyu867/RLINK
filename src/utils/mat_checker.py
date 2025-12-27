import os, glob
import numpy as np
import scipy.io as sio
from datetime import datetime, timedelta

def try_convert_labviewstart(x):
    """
    LabviewStart が何基準か不明なので、ありがちな変換候補をいくつか試して
    'それっぽい' datetime を返す（当たりを見つけるためのデバッグ用）。
    """
    if x is None:
        return []
    try:
        v = float(np.array(x).squeeze())
    except Exception:
        return []

    candidates = []

    # 1) MATLAB datenum (days since 0000-01-00) の可能性
    # MATLAB datenum 719529 = 1970-01-01
    # dt = datetime(1970,1,1) + timedelta(days=v-719529)
    try:
        dt = datetime(1970, 1, 1) + timedelta(days=v - 719529)
        candidates.append(("matlab_datenum", dt))
    except Exception:
        pass

    # 2) Excel serial date (days since 1899-12-30) の可能性
    try:
        dt = datetime(1899, 12, 30) + timedelta(days=v)
        candidates.append(("excel_serial", dt))
    except Exception:
        pass

    # 3) seconds since epoch (1970) の可能性
    try:
        dt = datetime(1970, 1, 1) + timedelta(seconds=v)
        candidates.append(("unix_seconds", dt))
    except Exception:
        pass

    # 4) LabVIEW timestamp (seconds since 1904-01-01) の可能性
    try:
        dt = datetime(1904, 1, 1) + timedelta(seconds=v)
        candidates.append(("labview_seconds_1904", dt))
    except Exception:
        pass

    return candidates

def peek_object(obj, name, max_depth=2, depth=0):
    """
    scipy.loadmat で読んだ object/cell/struct を浅く覗く。
    """
    indent = "  " * depth
    arr = np.array(obj)
    print(f"{indent}- {name}: dtype={arr.dtype}, shape={arr.shape}")

    if depth >= max_depth:
        return

    # MATLAB cell は dtype=object の nd-array になることが多い
    if arr.dtype == object and arr.size > 0:
        # 先頭数個だけ見る
        flat = arr.ravel()
        for i in range(min(3, flat.size)):
            item = flat[i]
            try:
                a = np.array(item)
                print(f"{indent}  * item[{i}]: type={type(item)}, asarray shape={a.shape}, dtype={a.dtype}")
                # structっぽい場合
                if hasattr(item, "_fieldnames"):
                    print(f"{indent}    fieldnames: {item._fieldnames}")
                # さらに1段だけ覗く
                if a.dtype == object and a.size > 0:
                    peek_object(item, f"{name}.item[{i}]", max_depth=max_depth, depth=depth+1)
            except Exception as e:
                print(f"{indent}  * item[{i}]: type={type(item)} (cannot arrayify) err={e}")

def inspect_expt(mat_path):
    print("="*90)
    print("[FILE]", mat_path)
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in data.keys() if not k.startswith("__")]
    print("Keys:", keys)

    # LabviewStart を候補変換
    if "LabviewStart" in data:
        lv = data["LabviewStart"]
        print("LabviewStart raw:", np.array(lv).squeeze())
        conv = try_convert_labviewstart(lv)
        for tag, dt in conv:
            # 極端に未来/過去は見にくいのでとりあえず表示
            print(f"  -> {tag}: {dt}")

    # Spike_data を覗く
    if "Spike_data" in data:
        sd = data["Spike_data"]
        peek_object(sd, "Spike_data", max_depth=3)

def main(data_dir, pattern="expt*.mat"):
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    print(f"Found {len(files)} files.")
    for f in files:
        inspect_expt(f)

if __name__ == "__main__":
    DATA_DIR = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/RL_decoders/datasets/original/monkey_1_set_1/expt"
    main(DATA_DIR, "expt*.mat")
