ui:
	streamlit run app.py

requirements:
	pipenv requirements > requirements.txt

up:
	docker-compose up

build:
	docker-compose build
