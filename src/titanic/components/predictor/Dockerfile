FROM kaggle/python:latest

RUN pip install kaggle
COPY src/ src/
RUN mkdir ~/.kaggle
COPY kaggle.json /root/.kaggle/kaggle.json

ENTRYPOINT ["python", "src/predictor.py"]
