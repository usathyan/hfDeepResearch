# hfDeepResearch

This project is for deep research using smolagents.

## Installation

You will need to set these ENV variables (and therefore API keys from external microservices)

* SERPAPI_API_KEY (Get it from serpapi)
* SMOL_KEY (openrouter.ai key - You can swap out the code to your model of choice)
* HF_TOKEN - if downloads from hf is needed

```bash
uv sync
```

## Usage

# Example usage (replace with your specific script and arguments)
```bash
python src/assistant.py --question "YHow long will it take a cheetah to go from Clinton, NJ to Lawrenceville, NJ"
```

Final answer: Given the detailed information provided, it does not appear that there is a specific, stated average speed 
for a cheetah in sustained runs over long distances. Cheetahs are known for their ability to reach speeds greater than 70 
miles per hour in short bursts, crucially needing to catch their prey within 30 seconds because they cannot maintain their 
maximum speed for longer durations. Given this, and the distance between Clinton, NJ, and Lawrenceville, NJ being 
approximately 31 miles, it would be unrealistic to estimate a specific travel time for a cheetah over this distance without
making broad and largely speculative assumptions about their sustained running speed over such a distance. Cheetahs' peak 
speeds and rapid acceleration are suited for short, high-speed chases rather than sustained long-distance travel. 
Therefore, calculating a precise time for a cheetah to travel between these two locations cannot be accurately determined 
with the available data.


## Note that i am using an openai key, that works for function calling. I am not happy with it yet, because the tool calling fails sometimes, GPT-4 is not the best model, and these are some of the limitations of the agentic framework.  It is not production ready.

