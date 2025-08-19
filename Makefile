.PHONY: example, start, build, client, docker_stop, docker_start, docker_remove, docker_open, docker_build

start:
	cd pie-cli && pie start --config ./example_config.toml

build:
	cd pie-cli && cargo install --path .
	cd backend/backend-python && chmod +x build_proto.sh
	cd backend/backend-python && ./build_proto.sh

example:
	cd example-apps && cargo build --target wasm32-wasip2 --release

client:
	cd client/python && python typego_s0.py

docker_stop:
	@echo "=> Stopping pie-typego..."
	@-docker stop -t 0 pie-typego > /dev/null 2>&1
	@-docker rm -f pie-typego > /dev/null 2>&

docker_start:
	@make docker_stop
	@echo "=> Starting pie-typego..."
	docker run -td --privileged --net=host \
    	--name="pie-typego" \
		--env-file ./docker/.env \
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