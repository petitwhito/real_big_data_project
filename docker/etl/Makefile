fast: Dockerfile
	tar --transform 's|^etl/||' -czvf apps.tgz etl/*
	docker build -t my_etl .

all: Dockerfile
	tar --transform 's|^etl/||' -czvf apps.tgz etl/*
	docker build --no-cache -t my_etl .

