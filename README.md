# SelectCode Mini-Hackathon

In diesem Repository findet ihr einen optionalen Startpunkt für unseren Mini-KI-Hackathon sowie einige weitere Links und Hilfestellungen.

## Wichtige Links
- [Slides](https://www.canva.com/design/DAGKAz5umZ0/KIYMoMBhzw_vOmQoNPv_WA/edit)
- [Meeting Transkripte](https://drive.google.com/drive/folders/1S7hJ5U6nGClbatBSZuaQpzsaA6Q2ElE3?usp=sharing)
- [API Key](https://bitwarden.selectcode.dev/#/send/cBLUj1wDSsWon9a_ojcoOg/o5wFGeBlDq1bpsptrXH99g)

Achtung: Der API-Key funktioniert nur mit unserem LLM-Proxy - du musst also die OPENAI_BASE_URL entsprechend setzen:

## In diesem Repo
Wenn du mit einem vorgefertigten Setup für eine Chat-Anwendung starten möchtest, enthält dieses Repository einen Startpunkt. Um zu starten:


```bash
# Clone the repo
git clone https://github.com/SelectCode/llm-example.git
```

Als nächstes, trag den OpenAI API Key in `.env` ein.

```bash
# Navigate into the repo
cd ./llm-example

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
# For Windows
env\Scripts\activate

# For Unix or MacOS
source env/bin/activate

# Install the dependencies from the `requirements.txt` file
pip install -r requirements.txt

# Launch chainlit
chainlit run app.py -w
```

### Los gehts!
Als nächstes kannst du damit starten, deine App anzupassen - schau dir z.B. `app.py` an!
In `utils.py` haben wir ein paar möglicherweise hilfreiche Methoden implementiert.

Auf der Suche nach Inspiration? Schau dir die [Chainlit Dokumentation](https://docs.chainlit.io/integrations/langchain) oder den [Überblick belieber LangChain Chains](https://python.langchain.com/docs/modules/chains/popular/) an.
Außerdem hilfreich: das [Chainlit Cookbook](https://github.com/Chainlit/cookbook/).

## Weitere Links
Einige weitere Ressourcen, die eventuell hilfreich sein könnten:
- [Streamlit (Python UI library für Data Apps)](https://streamlit.io/)
- [Vercel AI SDK (Tooling für Text-Streaming,...)](https://sdk.vercel.ai/docs/introduction)
- [Cohere Dokumentation (gute Tutorials,...)](https://docs.cohere.com/)
- [LlamaIndex (LLM data framework)](https://www.llamaindex.ai/)
- [Ragas (RAG Evaluation Tool)](https://docs.ragas.io/en/stable/)
- [Unstructured (Library für unstrukturierte Daten)](https://github.com/Unstructured-IO/unstructured)
- [Azure AI Services (breites Angebot an KI-Diensten)](https://azure.microsoft.com/de-de/products/ai-services)
