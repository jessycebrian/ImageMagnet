# ImageMagnet
Project using [OPEN AI's CLIP model](https://openai.com/research/clip) for text-to-image retrieval task.

`python pipeline.py` to run pipeline from reading dataset to upserting vectors to vector database in pinecone.

`streamlit run demo.py` to locally run ImageMagnet demo. You will need API_KEY and HOST from Pinecone to be able to access.

There is also a deployed demo version for access without having to run locally:
[ImageMagnet Demo](https://imagemagnet.streamlit.app/).


## Evaluation
### MRR
MRR for current implementation is
0.39 

