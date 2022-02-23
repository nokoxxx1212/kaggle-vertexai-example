FROM kaggle/python:latest

RUN pip install kaggle
COPY src/ src/

ENTRYPOINT ["python", "src/predictor.py"]