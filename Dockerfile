FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements-lock.txt ./
RUN python -c "import tomllib, subprocess; deps=tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; subprocess.check_call(['pip','install','--no-cache-dir','-c','requirements-lock.txt',*deps])"
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY configs ./configs
COPY artifacts ./artifacts
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,json; r=json.load(urllib.request.urlopen('http://localhost:8000/health')); assert r['model_ready']"
CMD ["uvicorn","predictmaint.service.app:app","--host","0.0.0.0","--port","8000"]
