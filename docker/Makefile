run:
	docker compose up

all:
	make -C etl
	make -C dashboard
	docker compose up

clean:
	docker compose rm
	sudo rm -rf ~/bourse/timescaledb/

pip:
	pipenv install

shell:
	pipenv shell
