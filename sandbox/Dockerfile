FROM python:3.11.15-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/work/mpl HOME=/work TMPDIR=/work PYTHONPATH=/opt/labweaver
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /opt/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /opt/requirements.lock
COPY runner.py helper.py /opt/labweaver/
RUN chmod -R a-w /opt/labweaver && mkdir /input /work && chown 65532:65532 /work
USER 65532:65532
WORKDIR /work
ENTRYPOINT ["python", "/opt/labweaver/runner.py"]
