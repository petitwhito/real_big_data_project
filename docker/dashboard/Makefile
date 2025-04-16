fast:
	tar --transform 's|^dashboard/||' -czvf apps.tgz dashboard/*
	docker build -t my_dashboard .

all: Dockerfile
	tar --transform 's|^dashboard/||' -czvf apps.tgz dashboard/*
	docker build --no-cache -t my_dashboard .

