import copy
import json
import os

from .utils.global_caption import CaptionStore, init_memory
from .utils.memory import create_memory_instance, delete_memory_files
from .utils.video_reader import FFmpegVideoReader
from .utils.openai_client import OpenAIClient
from .utils.tools import StopException, video_inspect_tool,memory_search_tool,finish
os.environ["OPENAI_API_KEY"] = "" # for embedder
os.environ["OPENAI_BASE_URL"] = ""
class Agent:
    def __init__(self, subset,instance_id,max_iterations,vlm_model,tool_call_model,memory_base):
        self.instance_id = instance_id
        self.memory_base = memory_base

        if subset == "stream":
            video_path = "./stream/merged.mp4"
        elif subset == "game":
            video_path = "./game/merged.mp4"
        elif subset == "egolife":
            video_path = "./egolife/merged.mp4"
        else:
            raise Exception
        lmdb_path = f"./cache/frame_lmdb/{subset}"
        self.video = FFmpegVideoReader(
            video_path=video_path,
            lmdb_root=lmdb_path,
            frames_per_shard=10000,
            jpeg_quality=75,
            ffmpeg_threads=16,
            fps_cached=2.0
        )
        self.openai_client = OpenAIClient(model_name=vlm_model['model_name'],api_key=vlm_model['api_key'],api_base=vlm_model['api_base'])
        self.openai_client_tool_call = OpenAIClient(model_name=tool_call_model['model_name'],api_key=tool_call_model['api_key'],api_base=tool_call_model['api_base'])
        self.openai_client_tool_call.register_tool(video_inspect_tool)
        self.openai_client_tool_call.register_tool(memory_search_tool)
        self.openai_client_tool_call.register_tool(finish)
        self.max_iterations = max_iterations
        self.messages = self._construct_messages()

        if not os.path.exists(f"./memory/{memory_base}"):
            caption_store = CaptionStore(path = f"./cache/caption/{subset}_{memory_base}.jsonl")
            init_memory(None,self.video,self.openai_client,caption_store = caption_store,clip_duration=300,num_workers=8)
            initial_memory = create_memory_instance(subset,instance_id,vlm_model,True,memory_base)
            caption_store.import_all_to_mem0(initial_memory)

        self.memory = create_memory_instance(subset,instance_id,vlm_model,True,memory_base)

    
    def clean_memory(self):
        delete_memory_files(self.instance_id,self.memory_base)

    def _construct_messages(self):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text":"""You are a helpful assistant who answers multi-step questions by sequentially invoking functions.
Follow the explicit THINK → ACT → OBSERVE loop.

For each step, you MUST explicitly output the following structured sections:

[REASONING]
Briefly and clearly explain your decision at a high level.
Do NOT reveal hidden chain-of-thought or token-level reasoning.
Summarize only the relevant considerations.

[ACTION]
Call exactly one function that moves you closer to the final answer,
or state that no function call is needed.

[OBSERVATION]
Summarize the result returned by the function call in a concise and factual manner.

You MUST plan before each function call and reflect on previous observations,
but your reasoning must be expressed only as a concise, human-readable summary.

Only pass arguments that come verbatim from the user or from earlier function outputs—never invent them.

Continue the loop until the user's query is fully resolved.
When finished, output the final answer or call `finish` if required.

If you are uncertain about code structure or video content, use the available tools
rather than guessing.

Timestamps may be formatted as 'HH:MM:SS'.
"""}
]            },
            {
                "role": "user",
                "content": [{"type": "text", "text":"""Carefully read the timestamps and visual descriptions retrieved during your analysis.
Pay close attention to the temporal and causal order of events, object attributes and movements,
and people’s actions and poses.

You may use the following tools whenever the available information is insufficient.

• To retrieve high-level and previously observed information about the video without specifying
  timestamps, use `memory_search_tool` if available. You should avoid calling `memory_search_tool` three times consecutively.

• If relevant time ranges are obtained from memory, or if no memory is available,
  you can use `video_inspect_tool` with a list of time ranges
  (list[tuple[HH:MM:SS, HH:MM:SS]]) to inspect the video clips in more detail.

• You may call `video_inspect_tool` multiple times with different or more focused time ranges
  as your understanding of the video improves.

• After you have gathered sufficient visual evidence, output the final answer using `finish`.
  Call `finish` only once.

Based on your observations and tool outputs, provide a concise answer that directly addresses
the question. If the available information is insufficient, thinking deeply and answer the question using general world knowledge.


Total video length: VIDEO_LENGTH seconds.

Question: QUESTION_PLACEHOLDER"""
                }]
            },
        ]
        video_length = f"{self.video.duration:.2f}"
        messages[-1]['content'][0]["text"] = messages[-1]['content'][0]["text"].replace("VIDEO_LENGTH", str(video_length))
        return messages
    
    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        msgs.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": name,
                "content": [{"type": "text", "text":content}],
            }
        )
    
    def _exec_tool(self, tool_call, msgs):
        name = tool_call.function.name
        if name not in self.openai_client_tool_call._tools:
            self._append_tool_msg(tool_call.id, name, f"Invalid function name: {name!r}", msgs)
            return

        # Parse arguments
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            raise StopException(f"Error decoding arguments: {exc!s}")

        # Call the tool
        try:
            print(f"Calling function `{name}` with args: {args}")
            result = self.openai_client_tool_call._tools[name](_video_db=self.video,_memory=self.memory,_openai_client=self.openai_client,**args)
            self._append_tool_msg(tool_call.id, name, result, msgs)
        except StopException as exc:  # graceful stop
            print(f"Finish task with message: '{exc!s}'")
            raise
    
    def run(self, question) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        """
        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"][0]["text"] = msgs[-1]["content"][0]["text"].replace("QUESTION_PLACEHOLDER", question)

        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration to avoid hanging
            if i == self.max_iterations - 1:
                msgs.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text":"Please call the `finish` function to finish the task. If the available information is insufficient, thinking deeply and answer the question using general world knowledge."}],
                    }
                )
            response = self.openai_client_tool_call.chat_with_tools(msgs,temperature=0.6,tool_choice="auto" if i != self.max_iterations -1 else {
    "type": "function",
    "function": { "name": "finish" }
  })

            print(f"#{i}",response)
            if response is None:
                return None

            msgs.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text":response["content"]}],
                        "tool_calls": response["tool_calls"]

                    })

            # Execute any requested tool calls
            try:
                for tool_call in response.get("tool_calls", []):
                    self._exec_tool(tool_call, msgs)
            except StopException:
                return msgs
            
        return msgs



if __name__ == "__main__":
    agent = Agent('stream','test',16,{'model_name':"Qwen/Qwen3-VL-30B-A3B-Instruct",'api_key':'EMPTY','api_base':'http://127.0.0.1:8000/v1'},{'model_name':"Qwen/Qwen3-VL-30B-A3B-Instruct",'api_key':'EMPTY','api_base':'http://127.0.0.1:8000/v1'})
    print(agent.run("During the live broadcast of Simon Charity Training, what was the reason IShowSpeed suddenly collapsed to the ground while walking and playing with his phone on the training ground?"))