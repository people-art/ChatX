# ChatX
Chat with pdf, word, images, audio, video files and so on ... you could chat with anything want. 

# How to usage
```bash
streamlit run app.py --server.port 8503
```


## Ingest Data
Put any and all your files into the `source_documents` directory

The supported extensions are:

- .csv: CSV,
- .docx: Word Document,
- .enex: EverNote,
- .eml: Email,
- .epub: EPub,
- .html: HTML File,
- .md: Markdown,
- .msg: Outlook Message,
- .odt: Open Document Text,
- .pdf: Portable Document Format (PDF),
- .pptx : PowerPoint Document,
- .txt: Text file (UTF-8),
Run the following command to ingest all the data.
```bash
python ingest.py
```

It will create a db folder containing the local vectorstore. Will take time, depending on the size of your documents. You can ingest as many documents as you want, and all will be accumulated in the local embeddings database. If you want to start from an empty database, delete the db folder.

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection.




