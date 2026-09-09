# Model is selected by ~/.pie/config.toml (this host: Qwen/Qwen3-0.6B on
# cuda_native). MODEL is recorded in the output rows only — see run.py.
MODEL ?= Qwen/Qwen3-0.6B
N_ROUNDS ?= 5
N_PARAGRAPHS ?= 6
ASKS_PER_ROUND ?= 5
CASES ?= 1,2,3,4,5,6
ROUNDS ?= :
MAX_TOKENS ?= 64

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: prepare run eval all smoke clean

prepare:
	cd $(HERE) && python prepare.py --n-rounds $(N_ROUNDS) \
		--n-paragraphs $(N_PARAGRAPHS) --asks-per-round $(ASKS_PER_ROUND)

run:
	python $(HERE)/run.py --model $(MODEL) --cases $(CASES) \
		--rounds $(ROUNDS) --max-tokens $(MAX_TOKENS)

eval:
	python $(HERE)/eval.py

all: prepare run eval

# Single round, all cases — quick end-to-end check.
smoke:
	python $(HERE)/run.py --model $(MODEL) --cases 1,2,3,4,5,6 \
		--rounds 0:1 --max-tokens $(MAX_TOKENS)

clean:
	rm -f $(HERE)/results.jsonl
