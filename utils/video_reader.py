import base64
import bisect
import os
import subprocess
import json
import tempfile
import numpy as np
import cv2
from tqdm import tqdm
import lmdb
from functools import lru_cache
from typing import Optional, List, Tuple, Iterator
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def _decode(buf):
    """从 LMDB bytes 解码为 numpy BGR 图"""
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

class VideoLMDB:
    def __init__(
        self,
        lmdb_root: str,
        frames_per_shard: int = 10000,
        jpeg_quality: int = 95,
    ):
        self.lmdb_root = lmdb_root
        self.frames_per_shard = frames_per_shard
        self.jpeg_quality = jpeg_quality
        os.makedirs(lmdb_root, exist_ok=True)

        # 缓存打开的 env
        self._open_env = lru_cache(maxsize=32)(self._open_env_impl)
        self.width = None
        self.height = None
        self.total_frames = None
        self._load_meta()

    def open_env(self, shard_id: int):
        """
        Open a specific shard LMDB (cached) and return the env.
        """
        return self._open_env(shard_id)

    def _load_meta(self):
        shard_files = sorted(f for f in os.listdir(self.lmdb_root) if f.startswith("shard_"))
        if not shard_files:
            return
        first_env = self._open_env(0)
        with first_env.begin() as txn:
            meta = json.loads(txn.get(b"__meta__").decode())
            self.width = meta["width"]
            self.height = meta["height"]
        self.total_frames = sum(
            int(lmdb.open(os.path.join(self.lmdb_root, f),
                         readonly=True, lock=False).begin().get(b"__len__"))
            for f in shard_files
        )

    def _open_env_impl(self, shard_id: int):
        path = os.path.join(self.lmdb_root, f"shard_{shard_id:05d}.lmdb")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)

    def get_frame(self, frame_id: int):
        shard_id = frame_id // self.frames_per_shard
        key = f"{frame_id:08d}".encode()
        env = self._open_env(shard_id)
        with env.begin(write=False) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"Frame {frame_id} not found")
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        return img

    def get_frames(self, frame_ids: List[int]):
        results = {}
        shard_groups = {}
        for fid in frame_ids:
            sid = fid // self.frames_per_shard
            shard_groups.setdefault(sid, []).append(fid)
        for sid, fids in shard_groups.items():
            env = self._open_env(sid)
            with env.begin(write=False) as txn:
                for fid in fids:
                    key = f"{fid:08d}".encode()
                    buf = txn.get(key)
                    if buf is None:
                        raise KeyError(fid)
                    results[fid] = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        return results

    def close(self):
        self._open_env.cache_clear()


class FFmpegVideoReader:
    def __init__(
        self,
        video_path: str,
        lmdb_root: str,
        frames_per_shard: int = 10000,
        jpeg_quality: int = 95,
        ffmpeg_threads: int = 16,
        fps_cached = None
    ):
        self.video_path = video_path
        self.lmdb_root = lmdb_root
        self.frames_per_shard = frames_per_shard
        self.jpeg_quality = jpeg_quality
        self.ffmpeg_threads = ffmpeg_threads

        # probe video meta
        self.meta = self._probe()
        video_stream = next(s for s in self.meta["streams"] if s["codec_type"] == "video")
        self.width = int(video_stream["width"])
        self.height = int(video_stream["height"])
        self.fps = self._get_fps(video_stream)
        self.duration = float(self.meta["format"]["duration"])
        self.num_frames = int(self.duration * self.fps)

        # 初始化 LMDB
        if not os.path.exists(lmdb_root) or not os.listdir(lmdb_root):
            print("[LMDB] Caching video frames to LMDB ...")
            self._cache_to_lmdb(target_fps=fps_cached)
        self.db = VideoLMDB(lmdb_root)
        if not hasattr(self, "fps_cached"):
            # 从第一个 shard meta 读取
            env = self.db._open_env(0)
            with env.begin(write=False) as txn:
                meta_bytes = txn.get(b"__meta__")
                if meta_bytes:
                    meta = json.loads(meta_bytes.decode())
                    self.fps_cached = meta.get("fps_cached", 1.0)
                else:
                    self.fps_cached = 1.0

    # ============================================================
    # Probe video
    # ============================================================
    def _probe(self):
        cmd = [
            "./ffmpeg/ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            self.video_path,
        ]
        out = subprocess.check_output(cmd)
        return json.loads(out)

    def _get_fps(self, video_stream):
        rate = video_stream.get("avg_frame_rate", "0/1")
        num, den = map(int, rate.split("/"))
        return num / den if den != 0 else 0.0

    # ============================================================
    # LMDB caching
    # ============================================================
    def _cache_to_lmdb(self, batch_size: int = 1000, target_fps: float = 1.0):
        """
        Cache entire video into LMDB at target_fps in batches.
        batch_size: number of frames to read at once from FFmpeg pipe
        """
        os.makedirs(self.lmdb_root, exist_ok=True)

        vf = f"fps={target_fps}"
        cmd = [
            "./ffmpeg/ffmpeg",
            "-loglevel", "error",  # 避免 info/warning 阻塞
            "-i", self.video_path,
            "-vf", vf,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]

        frame_size = self.width * self.height * 3
        read_size = frame_size * batch_size

        shard_count = 0
        frame_id = 0
        shard_id = 0
        env = None
        txn = None

        def new_shard():
            nonlocal env, txn, shard_count, shard_id
            if env:
                txn.put(b"__len__", str(shard_count).encode())
                txn.commit()
                env.sync()
                env.close()
            path = os.path.join(self.lmdb_root, f"shard_{shard_id:05d}.lmdb")
            env = lmdb.open(
                path,
                map_size=200 * 1024**3,
                subdir=True,
                meminit=False,
                map_async=True,
            )
            txn = env.begin(write=True)
            meta = {
                "width": self.width,
                "height": self.height,
                "shard_id": shard_id,
                "frames_per_shard": self.frames_per_shard,
                "fps_cached": target_fps,
            }
            txn.put(b"__meta__", json.dumps(meta).encode())
            shard_count = 0
            shard_id += 1

        new_shard()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )

        total_frames_estimate = int(self.duration * target_fps)
        pbar = tqdm(total=total_frames_estimate, desc=f"Caching {target_fps}FPS frames to LMDB")

        while True:
            raw = proc.stdout.read(read_size)
            if len(raw) == 0:
                break
            n_frames = len(raw) // frame_size
            frames = np.frombuffer(raw[:n_frames * frame_size], np.uint8).reshape(
                n_frames, self.height, self.width, 3
            )

            for i in range(n_frames):
                frame = frames[i]
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ok:
                    continue
                if shard_count >= self.frames_per_shard:
                    new_shard()
                txn.put(f"{frame_id:08d}".encode(), buf.tobytes())
                shard_count += 1
                frame_id += 1

            pbar.update(n_frames)

            # 每 500 帧提交一次事务
            if shard_count % 500 == 0:
                txn.commit()
                txn = env.begin(write=True)

        # 最后提交事务
        txn.put(b"__len__", str(shard_count).encode())
        txn.commit()
        env.sync()
        env.close()
        proc.stdout.close()
        proc.wait()
        pbar.close()
        print(f"[LMDB Cache] Done. Total frames cached: {frame_id}")
        self.fps_cached = target_fps
        self.total_frames = frame_id

    # ============================================================
    # Sample frames by FPS
    # ============================================================
    def sample_frames_by_fps(
        self,
        start_sec: float,
        end_sec: float,
        target_fps: float,
        max_frames: int,
        max_workers: int = 8
    ):
        """
        优化版：利用 set_range 查找最近邻，避免全表扫描。
        """
        fps_cached = getattr(self, "fps_cached", 1.0)
        start_idx = int(start_sec * fps_cached)
        end_idx = min(int(end_sec * fps_cached), self.db.total_frames - 1)
        total_frames = end_idx - start_idx + 1
        if total_frames <= 0:
            return [], [], []

        total_seconds = end_sec - start_sec
        ideal_step_sec = total_seconds / max_frames
        step = max(1, int(ideal_step_sec * fps_cached))
        requested_indices = list(range(start_idx, end_idx + 1, step))
        requested_indices = requested_indices[:max_frames]

        # 分 shard
        shard_groups = {}
        for fid in requested_indices:
            sid = fid // self.db.frames_per_shard
            shard_groups.setdefault(sid, []).append(fid)

        results = {}
        # 进度条
        pbar = tqdm(total=len(requested_indices), desc="Reading LMDB frames", unit="frame", leave=False)
        pbar_lock = threading.Lock()

        def _load_shard(sid, fids):
            out = {}
            try:
                env = self.db.open_env(sid)
                # 使用 buffers=True 减少内存拷贝（如果 _decode 支持 buffer，否则这行可不用）
                with env.begin(write=False, buffers=True) as txn:
                    cursor = txn.cursor()
                    
                    for fid in fids:
                        target_key_str = f"{fid:08d}"
                        target_key = target_key_str.encode()
                        
                        # 1. 尝试直接获取
                        buf = txn.get(target_key)
                        
                        # 2. 如果不存在，使用 set_range 寻找最近的帧
                        if buf is None:
                            # set_range 会定位到 >= target_key 的第一个位置
                            found = cursor.set_range(target_key)
                            
                            candidates = []
                            
                            # 获取 "后一个" (set_range 定位的位置)
                            if found:
                                k_after, v_after = cursor.item()
                                if k_after not in (b"__meta__", b"__len__"):
                                    try:
                                        k_idx_after = int(k_after)
                                        candidates.append((k_idx_after, v_after))
                                    except ValueError:
                                        pass
                            
                            # 获取 "前一个"
                            # 注意：如果刚才 set_range 成功了，光标在 after 位置，prev() 会回到前一个
                            # 如果 set_range 失败（比如 fid 比所有 key 都大），光标位置未定义，需要处理
                            if found:
                                if cursor.prev():
                                    k_before, v_before = cursor.item()
                                    if k_before not in (b"__meta__", b"__len__"):
                                        try:
                                            k_idx_before = int(k_before)
                                            candidates.append((k_idx_before, v_before))
                                        except ValueError:
                                            pass
                                    # 恢复光标位置以便逻辑清晰（可选，因为下次循环会重新 set_range）
                            else:
                                # set_range 失败说明没有比 target 更大的，尝试定位到最后
                                if cursor.last():
                                    k_last, v_last = cursor.item()
                                    if k_last not in (b"__meta__", b"__len__"):
                                         try:
                                            candidates.append((int(k_last), v_last))
                                         except ValueError:
                                            pass

                            # 比较找出最近的
                            if candidates:
                                nearest_idx, nearest_buf = min(candidates, key=lambda x: abs(x[0] - fid))
                                buf = nearest_buf
                            else:
                                # 这个 shard 可能是空的或者只有 meta 信息
                                buf = None

                        if buf is not None:
                            # 假设 _decode 是定义的解码函数
                            # 如果 buf 是 memoryview/buffer，有些解码库可能需要 bytes(buf)
                            out[fid] = _decode(buf)
                            with pbar_lock:
                                pbar.update(1)
                                
            except Exception as e:
                print(f"Error reading shard {sid}: {e}")
                import traceback
                traceback.print_exc()
            return out

        # 启动线程池
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_load_shard, sid, fids) for sid, fids in shard_groups.items()]
            for fut in as_completed(futures):
                results.update(fut.result())

        pbar.close()

        # 输出处理（保持原有逻辑）
        frames_list = []
        timestamps = []
        actual_indices = []

        for fid in requested_indices:
            if fid in results:
                frames_list.append(results[fid])
                timestamps.append((fid-start_idx) / fps_cached)
                actual_indices.append(fid)
            elif results:
                # 只有当 _load_shard 完全没找到数据时才会走到这里
                # (通常 _load_shard 内部已经处理了最近邻逻辑)
                nearest = min(results.keys(), key=lambda x: abs(x - fid))
                frames_list.append(results[nearest])
                timestamps.append((fid-start_idx) / fps_cached)
                actual_indices.append(nearest)

        return frames_list, timestamps, actual_indices

    def cut_segment(
        self,
        start_sec: float,
        end_sec: float,
        keyframe_seek: bool = True,
    ) -> str:
        """
        Cut a segment from video and return its content as base64 string.

        Args:
            video_path: source video file path
            start_sec: start time in seconds
            end_sec: end time in seconds
            keyframe_seek: whether to seek to keyframe first

        Returns:
            Base64-encoded bytes of the segment (MP4)
        """
        assert end_sec > start_sec
        duration = end_sec - start_sec

        # 创建临时文件存放裁剪结果
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpfile:
            cmd = ["./ffmpeg/ffmpeg", "-y"]

            if keyframe_seek:
                cmd += ["-ss", str(start_sec)]

            cmd += ["-i", self.video_path]

            if not keyframe_seek:
                cmd += ["-ss", str(start_sec)]

            cmd += [
                "-t", str(duration),
                "-map", "0",
                "-c", "copy",
                "-movflags", "+faststart",
                tmpfile.name,
            ]

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # 读取文件内容并转成 base64
            tmpfile.seek(0)
            data = tmpfile.read()
            return base64.b64encode(data).decode("utf-8")
    
    def cut_segment_to_audio(
        self,
        start_sec: float,
        end_sec: float,
        keyframe_seek: bool = True,
    ) -> str:
        """
        Cut a segment of audio from video and return its content as base64 string (WAV).

        Args:
            start_sec: start time in seconds
            end_sec: end time in seconds
            keyframe_seek: whether to seek before decoding (faster but less precise)

        Returns:
            Base64-encoded bytes of the segment in WAV format
        """
        assert end_sec > start_sec
        duration = end_sec - start_sec

        # 创建临时文件存放裁剪结果
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            cmd = ["./ffmpeg/ffmpeg", "-y"]

            # 如果 keyframe_seek，先快速跳转
            if keyframe_seek:
                cmd += ["-ss", str(start_sec)]

            cmd += ["-i", self.video_path]

            # 如果不是 keyframe_seek，跳转放在输入后，保证精确
            if not keyframe_seek:
                cmd += ["-ss", str(start_sec)]

            cmd += [
                "-t", str(duration),
                "-vn",             # 不处理视频，只提取音频
                "-acodec", "pcm_s16le",  # WAV 默认编码
                "-ar", "44100",         # 可选：采样率
                "-ac", "2",             # 可选：立体声
                tmpfile.name
            ]

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # 读取文件内容并转成 base64
            tmpfile.seek(0)
            data = tmpfile.read()
            return base64.b64encode(data).decode("utf-8")