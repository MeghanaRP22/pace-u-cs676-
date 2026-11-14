# Container image for PersonaForge: the Streamlit web app by default, the
# terminal app on request.
#
#   docker build -t personaforge .
#   docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... personaforge     # web, live
#   docker run -p 8501:8501 personaforge                                    # web, offline
#   docker run -it --env-file .env personaforge python main.py              # terminal
#
# The key is passed at run time and never baked into the image.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so editing source does not reinstall them on rebuild.
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install --no-cache-dir -r deploy/requirements.txt

COPY src src
COPY deploy deploy
COPY main.py .

# The package is not pip-installed; put it on the path, as the web app does.
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["streamlit", "run", "deploy/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
