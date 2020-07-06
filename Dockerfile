FROM python:3.8

MAINTAINER Sanjula Madurapperuma "developer.sanjula@gmail.com"

ADD src/StackOverflowTagsUpdated.csv /

ADD src/my_bow.csv /

ADD src/requirements.txt /

ADD src/TrendAnalysis.py /

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "./TrendAnalysis.py"]