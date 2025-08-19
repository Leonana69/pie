.PHONY: example, start, build, typego, docker_stop, docker_start, docker_remove, docker_open, docker_build

start:
	cd pie-cli && pie start --config ./example_config.toml

build:
	cd pie-cli && cargo install --path .

example:
	cd example-apps && cargo build --target wasm32-wasip2 --release

typego:
	cd client/python && python typego_s0.py

GPU_DEVICES=5
GPU_OPTIONS=$(shell if [ -f /proc/driver/nvidia/version ]; then echo "--gpus all -e CUDA_VISIBLE_DEVICES=$(GPU_DEVICES)"; else echo ""; fi)

docker_stop:
	@echo "=> Stopping pie-typego..."
	@-docker stop -t 0 pie-typego > /dev/null 2>&1
	@-docker rm -f pie-typego > /dev/null 2>&1

docker_start:
	@make docker_stop
	@echo "=> Starting pie-typego..."
	docker run -td --privileged --net=host $(GPU_OPTIONS) \
    	--name="pie-typego" \
		--env-file ./docker/.env \
		-v ~/.cache/pie:/root/.cache/pie \
		pie-typego:0.1

docker_remove:
	@echo "=> Removing pie-typego..."
	@-docker image rm -f pie-typego:0.1  > /dev/null 2>&1
	@-docker rm -f pie-typego > /dev/null 2>&1

docker_open:
	@echo "=> Opening bash in pie-typego..."
	@docker exec -it pie-typego bash

docker_build:
	@echo "=> Building pie-typego..."
	@make docker_stop
	@make docker_remove
	@echo -n "=>"
	docker build -t pie-typego:0.1 -f ./docker/Dockerfile .
	@echo -n "=>"
	@make docker_start