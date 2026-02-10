from concurrent.futures import ThreadPoolExecutor
import json
import threading
from queue import Queue
from math import ceil

import tqdm
from .helper import image_to_base64, timesec_hms

CAPTION_PROMPT = """
You are a multimodal video understanding assistant. Generate a detailed caption for the given video clip.

Requirements:
1. Analyze the visual information, including actions, expressions, scene elements, objects, and people.
2. Describe any visible text in the video (subtitles, signs, etc.).
3. Include absolute timestamps [HH:MM:SS] at key actions, changes, or events, at the start of the sentence or segment.
   - Only mark the most significant moments, with a maximum of 10 timestamps.
4. Use natural language, at least one sentence per segment, and avoid repeating information.
5. Do not speculate; describe only what is directly observable.

Provide the final caption with absolute timestamps at the most important points.
"""


import threading
import json
import os

class CaptionStore:
    def __init__(self, path):
        """
        path: 保存 caption 的 JSONL 文件路径
        """
        self.path = path
        self.lock = threading.Lock()
        self.existing_ranges = set()  # (start,end) 对
        # 如果文件存在，读取已有记录
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        self.existing_ranges.add((item["start_time"], item["end_time"]))
                    except Exception:
                        continue  # 忽略损坏行

    def add(self, start, end, caption):
        """
        添加一个 caption，如果已经存在则跳过
        """
        key = (start, end)
        with self.lock:
            if key in self.existing_ranges:
                # 已经存在，跳过
                return False
            # 写入文件
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "start_time": start,
                    "end_time": end,
                    "caption": caption
                }, ensure_ascii=False) + "\n")
            # 更新索引
            self.existing_ranges.add(key)
            return True

    def has(self, start, end):
        """
        检查这个时间段是否已经存在
        """
        with self.lock:
            return (start, end) in self.existing_ranges
    
    def import_all_to_mem0(
        self,
        memory,
        user_id: str = "root",
        mem_type: str = "video_info",
        max_workers: int = 8,
    ):
        def process_line(line: str):
            if not line.strip():
                return

            try:
                item = json.loads(line)
            except Exception:
                return

            start = item["start_time"]
            end = item["end_time"]
            caption = item["caption"]

            memory.add(
                [
                    {
                        "role": "user",
                        "content": (
                            f"From {timesec_hms(start)} "
                            f"to {timesec_hms(end)}, {caption}"
                        ),
                    }
                ],
                user_id=user_id,
                metadata={
                    "start_time": start,
                    "end_time": end,
                    "mem_type": mem_type,
                },
                infer=False,
            )

        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                tqdm.tqdm(
                    executor.map(process_line, lines),
                    total=len(lines),
                )
            )

def init_memory(memory, video, openai_client, clip_duration=300, max_queue_size=16, num_workers=8,caption_store=None):
    num_clips = ceil(video.duration / clip_duration)
    clip_queue = Queue(maxsize=max_queue_size)
    stop_signal = object()

    def producer():
        for clip_idx in range(num_clips):
            start_time_sec = clip_idx * clip_duration
            end_time_sec = min((clip_idx + 1) * clip_duration, video.duration)
            if caption_store.has(start_time_sec,end_time_sec):
                continue
            # audio_b64 = video.cut_segment_to_audio(start_time_sec, end_time_sec)
            frames_list, timestamps, actual_indices = video.sample_frames_by_fps(start_time_sec,end_time_sec,1,300)

            video_context = []
            
            def _proc_frame(args):
                frame,timestamp = args
                img64 = image_to_base64(frame, (640, 360), 75)
                return [
                    {"type": "text", "text": f"[{timesec_hms(timestamp)}]"},
                    {"type": "image_url", "image_url": {"url": img64}}
                ]

            print("Encoding frames to Base64...")
            with ThreadPoolExecutor(max_workers=16) as pool:
                # 使用 map 保持顺序，或者使用 as_completed 后再排序
                results = list(tqdm.tqdm(pool.map(_proc_frame, zip(frames_list,timestamps)), total=len(frames_list)))
                for res in results:
                    video_context.extend(res)
            
            clip_queue.put((start_time_sec, end_time_sec, video_context, None))
        
        for _ in range(num_workers):
            clip_queue.put(stop_signal)


    def consumer():
        while True:
            item = clip_queue.get()
            if item is stop_signal:
                break
            
            start_time_sec, end_time_sec, video_context,audio_b64 = item

            print(start_time_sec,end_time_sec)
            
            caption = openai_client.chat(messages=[{
                "role": "user",
                "content": video_context+[
                    # {
                    #     "type": "input_audio",
                    #     "input_audio": {
                    #         "data": audio_b64,
                    #         "format": "wav",
                    #     },
                    # },
                    {"type": "text", "text": CAPTION_PROMPT},
                ]
            }])

            caption = openai_client.chat(messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
                    You are given:
                    1) A block of text that may contain multiple timestamps in the format [HH:MM:SS]
                    2) A time offset in the format HH:MM:SS

                    Task:
                    - Shift EVERY timestamp in the text by the given offset.
                    - A timestamp [HH:MM:SS] represents a time duration, not a clock time.
                    - The offset should be ADDED to each timestamp.
                    - Properly handle carry-over for seconds and minutes.
                    - Preserve the original [HH:MM:SS] format (always two digits per field).
                    - Do NOT modify any part of the text other than the timestamps.
                    - Do NOT add, remove, or rephrase any text.

                    If the text contains no timestamps, return the original text unchanged.

                    Text:
                    {caption}

                    Time offset:
                    {timesec_hms(start_time_sec)}
                    
                    Output only the modified text. Do not include any other content.
                    """
                    },
                ]
            }])

            print(caption)

            caption_store.add(
                start_time_sec,
                end_time_sec,
                caption
            )
            if memory:
                memory.add(
                    [{
                        "role": "user",
                        "content": f"From {timesec_hms(start_time_sec)} to {timesec_hms(end_time_sec)}, {caption}"
                    }],
                    user_id="root",
                    metadata={"start_time": start_time_sec, "end_time": end_time_sec,"mem_type": "video_info"},
                    infer=False
                )

    prod_thread = threading.Thread(target=producer)
    prod_thread.start()

    consumer_threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=consumer)
        t.start()
        consumer_threads.append(t)

    prod_thread.join()
    for t in consumer_threads:
        t.join()



