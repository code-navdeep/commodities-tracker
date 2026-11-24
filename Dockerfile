FROM python:3.13-slim

WORKDIR /app

# No apt packages needed: no kaleido/chrome (charts render in the browser),
# and every wheel in the lock installs clean on slim.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY app.py pyproject.toml ./
COPY commodities/ commodities/
COPY assets/ assets/

EXPOSE 8057

CMD ["python3", "app.py"]
