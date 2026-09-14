FROM python:3.12-slim
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /opt/qc
COPY qc qc
COPY tests tests
# Apptainer starts in the host's working directory, so imports can't rely on WORKDIR.
ENV PYTHONPATH=/opt/qc
ENTRYPOINT ["python", "-m", "qc.run"]
