.PHONY: generate
generate:
	mkdir -p generated
	python -m grpc_tools.protoc -I. --python_out=generated --grpc_python_out=generated api/**/*.proto
