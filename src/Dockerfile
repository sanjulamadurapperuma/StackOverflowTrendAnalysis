FROM python:3.8

MAINTAINER Sanjula Madurapperuma "developer.sanjula@gmail.com"

ADD StackOverflowTagsUpdated.csv /

ADD my_bow.csv /

ADD requirements.txt /

ADD TrendAnalysis.py /

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "./TrendAnalysis.py"]