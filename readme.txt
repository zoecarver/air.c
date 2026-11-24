AIR(1)                      General Commands Manual                     AIR(1)

NAME
     air - Inkling 2 Mini on Apple silicon

SYNOPSIS
     air prep checkpoint out.air
     air chat [-m model.air] [-c ctx] [-e effort]
     air serve [-m model.air] [-c ctx] [-e effort] [-h host] [-p port]
               [-d cachedir] [-H]

DESCRIPTION
     Runs Thinking Machines' Inkling 2 Mini on the GPU with 4-bit weights and
     speculative decoding from the model's own drafter. Greedy only.

     prep   pack the Hugging Face checkpoint into one 16 GB file
     chat   chat in the terminal; /reset, /quit
     serve  Anthropic Messages API on /v1/messages (streaming, tools,
            thinking). Long shared prompt prefixes are checkpointed to
            cachedir.

     -m     weights (inkling.air)
     -c     context length (65536)
     -e     thinking effort, 0 to 0.99 (0.1); 0.7 when a request enables
            thinking
     -h -p  listen address (127.0.0.1 8080)
     -d     checkpoint directory (cache)
     -H     run requests for haiku models instead of answering "OK"

EXAMPLES
     make
     hf download thinkingmachines/Inkling-2-Mini --local-dir Inkling-2-Mini
     ./air prep Inkling-2-Mini inkling.air
     ./air chat

     ./air serve
     ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=x \
     ANTHROPIC_MODEL=inkling-2-mini CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 claude

PERFORMANCE
     Tokens per second on a MacBook Air M4 (10-core GPU, 32 GB), fresh
     prompts, greedy.

                                air    MTPLX 2.13   llama.cpp
     decode                      25        17           14
     decode at 54k context       20        12
     prefill, 1k tokens         190       133          127
     prefill, 10k tokens        196       125
     prefill, 54k tokens        129       116

     Needs about 20 GB of free memory and macOS 15.

     I made this for myself over the weekend and do not guarantee quality or
     stability; you can file GitHub issues or PRs, but I likely won't look at
     them.
