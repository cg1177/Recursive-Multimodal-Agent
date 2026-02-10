from concurrent.futures import ThreadPoolExecutor
import os
from queue import Queue
import threading
from typing import Annotated as A

from tqdm import tqdm
from .func_call_shema import doc as D
from .helper import image_to_base64, timesec_hms
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

SUMMARIZE_WORKERS = 8

SEARCH_WORKERS = 8

VIDEO_GROUNDING_PROMPT = """
You are given a video.
Your task is to identify the time ranges in the video that are relevant to the user query.
Instructions:
- Carefully watch the video and consider both visual and audio information.
- Identify video segments where the content directly matches the user query.
- Each selected segment should be a continuous time range.
- If multiple relevant segments exist, list all of them.
- For each segment, provide a brief reason or description for why this segment was selected.
- If no relevant segment is found, output NOT FOUND.
- Do NOT describe unrelated parts of the video.

User query:
"{query}"

Output format:
HH:MM:SS - HH:MM:SS, <reason or description>
(one line per segment; if nothing is found, output only NOT FOUND)
"""

def video_inspect_tool(
    _video_db,
    _memory,
    _openai_client,
    question: A[str, D("The specific detailed question to ask about the video content during the specified time ranges. No need to add time ranges in the question.")], 
    time_ranges_hhmmss: A[list[tuple], D("A list of tuples containing start and end times in HH:MM:SS format, at most 10 intervals.")], 
) -> str:
    """
    Crop the video based on the time ranges and ask the model a detailed question about the cropped video clips.
    Returns:
        str: The model's response to the question. If no relevant content is found within the time range,
             returns an error message: "Error: Cannot find corresponding result in the given time range."
    """

    clip_queue = Queue(maxsize=16)
    stop_signal = object()

    results = []
    results_lock = threading.Lock()

    def producer():
        for s_hms, e_hms in time_ranges_hhmmss:
            start_time_sec,end_time_sec = timesec_hms(s_hms,out="float"),min(timesec_hms(e_hms,out="float"),_video_db.duration)
            # audio_b64 = _video_db.cut_segment_to_audio(start_time_sec, end_time_sec)
            frames_list, timestamps, actual_indices = _video_db.sample_frames_by_fps(start_time_sec,end_time_sec,2,300 if "gpt" not in _openai_client.model_name else 50)
            
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
                results = list(tqdm(pool.map(_proc_frame, zip(frames_list,timestamps)), total=len(frames_list)))
                for res in results:
                    video_context.extend(res)
            
            clip_queue.put((start_time_sec, end_time_sec, video_context, None))
        
        for _ in range(8):
            clip_queue.put(stop_signal)


    def consumer():
        while True:
            item = clip_queue.get()
            if item is stop_signal:
                break
            
            start_time_sec, end_time_sec, video_context,audio_b64 = item
            
            caption = _openai_client.chat(messages=[{
                "role": "user",
                "content": 
                    video_context+[
                    # {
                    #     "type": "input_audio",
                    #     "input_audio": {
                    #         "data": audio_b64,
                    #         "format": "wav",
                    #     },
                    # },
                    {
                        "type": "text",
                        "text": (
                            "Carefully watch the video. Pay close attention to the cause and sequence of events, "
                            "the details and movements of objects, and the actions and poses of people.\n\n"
                            "Based on your observations, answer the question using only information that can be "
                            "directly verified from the video.\n\n"
                            "When relevant, you MAY insert time anchors from the video into your answer "
                            "to support your reasoning. Time anchors must be in the format [HH:MM:SS] and should "
                            "correspond exactly to the moment shown in the video.\n\n"
                            "Do NOT invent timestamps. If you are uncertain about the exact time, omit the time anchor.\n\n"
                            "If no relevant content is found within the given time range, return exactly:\n"
                            "`Error: Cannot find corresponding result in the given time range.`\n\n"
                            f"Question: {question}\n"
                        )
                        },

                ]
            }])

            caption = _openai_client.chat(messages=[{
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

            _memory.add(
                {
                    "role": "user",
                    "content": (
                        f"From {timesec_hms(start_time_sec)} to {timesec_hms(end_time_sec)}, "
                        f"answering the question '{question}', the video clip shows: {caption}"
                    )
                },
                user_id="root",
                metadata={
                    "start_time": start_time_sec,
                    "end_time": end_time_sec,
                    "question": question,
                    "mem_type": "video_info"
                }
            )

            # 线程安全地把结果 append 到列表
            with results_lock:
                results.append({
                    "start_time": start_time_sec,
                    "end_time": end_time_sec,
                    "caption": caption
                })
        
    prod_thread = threading.Thread(target=producer)
    prod_thread.start()

    consumer_threads = []
    for _ in range(8):
        t = threading.Thread(target=consumer)
        t.start()
        consumer_threads.append(t)

    prod_thread.join()
    for t in consumer_threads:
        t.join()
    
    return "\n".join([f"From {timesec_hms(msg['start_time'])} to {timesec_hms(msg['end_time'])}, {msg['caption']}" for msg in results])

def _search_single_query(_memory, q, top_k):
    result = _memory.search(
        q,
        user_id="root",
        filters={"mem_type": "video_info"},
        limit=top_k,
    )

    video_hits = []
    contextual = []

    for item in result.get("results", []):
        metadata = item.get("metadata", {}) or {}
        mem = item.get("memory", "").replace("\n", " ").strip()

        st = metadata.get("start_time")
        ed = metadata.get("end_time")

        if st is not None and ed is not None and mem:
            video_hits.append((st, ed, q, mem))
        elif mem:
            contextual.append(mem)

    return video_hits, contextual

def _summarize_and_add(
    _openai_client,
    _memory,
    start_time,
    end_time,
    q,
    summarize_query,
    texts,
):
    merged = " ".join(texts)

    summary = summarize_with_gpt(
        _openai_client=_openai_client,
        query=q,
        summarize_query=summarize_query,
        text=merged,
    )

    if not summary or "NOT FOUND" in summary.strip():
        return None

    line = (
        f"From {timesec_hms(start_time)} to {timesec_hms(end_time)}, "
        f"(query: {q}) {summary}"
    )

    _memory.add(
        {
            "role": "user",
            "content": (
                f"From {timesec_hms(start_time)} to {timesec_hms(end_time)}, "
                f"related to query '{q}', {summary}"
            ),
        },
        user_id="root",
        metadata={
            "start_time": start_time,
            "end_time": end_time,
            "question": q,
            "mem_type": "video_info",
        }
    )

    return line

def memory_search_tool(
    _video_db,
    _memory,
    _openai_client,
    query: A[str, D("Short text queries for memory search; use ';' to separate up to 5 queries.")],
    summarize_query: A[
        str,
        D("Query used during LLM summarization to filter and extract only useful information.")
    ],
    top_k: A[int, D("Maximum number of relevant memory to return. Default is 10.")] = 10,
) -> str:
    """
    Retrieve relevant video memory using textual search queries and perform
    query-conditioned summarization to extract task-relevant evidence by LLM.

    This function first retrieves candidate memory entries based on one or more
    search queries. It then applies a separate summarization query to filter out
    irrelevant content and generate concise, question-oriented summaries.

    Returns:
        str:
            A newline-separated list of video evidence in the format:
            "From HH:MM:SS to HH:MM:SS, <summary>".

            Only summaries that are relevant to the summarization query are included.
            If no relevant evidence is found, an empty string is returned.
    """

    queries = [q.strip() for q in query.split(";") if q.strip()]

    if not queries:
        return ""

    video_grouped = defaultdict(list)
    contextual_memories = []

    # ===== 1. parallel search =====
    with ThreadPoolExecutor(
        max_workers=min(SEARCH_WORKERS, len(queries))
    ) as executor:

        futures = [
            executor.submit(_search_single_query, _memory, q, top_k)
            for q in queries
        ]

        for fut in as_completed(futures):
            hits, context = fut.result()
            contextual_memories.extend(context)

            for st, ed, q, mem in hits:
                video_grouped[(st, ed, q)].append(mem)

    if not video_grouped and not contextual_memories:
        return ""

    sections = []
    if video_grouped:
        sections.append("[Video Evidence]")

    # ===== 2. parallel summarize + add =====
    with ThreadPoolExecutor(max_workers=SUMMARIZE_WORKERS) as executor:
        sum_futures = []

        for (st, ed, search_q), texts in video_grouped.items():
            sum_futures.append(
                executor.submit(
                    _summarize_and_add,
                    _openai_client,
                    _memory,
                    st,
                    ed,
                    search_q,
                    summarize_query,
                    texts,
                )
            )

        for fut in as_completed(sum_futures):
            line = fut.result()
            if line:
                sections.append(line)

    # ===== 3. contextual memory =====
    if contextual_memories:
        sections.append("\n[Contextual Memory]")
        sections.extend(contextual_memories)

    return "\n".join(sections)
    
def summarize_with_gpt(_openai_client, query: str,summarize_query, text: str) -> str:
    prompt = f"""
        You are summarizing retrieved video memory.

        Search query (for retrieval):
        {query}

        Filtering / summarization query (IMPORTANT):
        {summarize_query}

        Below are memory snippets retrieved from the same video segment.
        Only keep information that is directly useful for answering the filtering query.

        Rules:
        - If the content does NOT help answer the filtering query, return an empty string.
        - Be concise and factual.
        - Do NOT speculate.
        - If useful, produce ONE concise sentence.

        Memory snippets:
        {text}
        """

    return _openai_client.chat(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ]
    }])


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


def finish(_video_db,
    _memory,
    _openai_client,answer: A[str, D("Answer to the user's question.")]) -> None:
    """Call this function after confirming the answer of the user's question, and finish the conversation."""
    raise StopException(answer)

