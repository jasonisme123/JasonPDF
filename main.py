from langchain.document_loaders import PyPDFLoader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from flask import Flask, request
import datetime
import sys
# from pympler import asizeof
from langchain.embeddings import FakeEmbeddings


app = Flask(__name__)
memorydb = None
chat_history = []  # 建议将聊天记录保存在客户的浏览器上


@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('index.html')


@app.route('/chat', methods=['GET'])
def chat():
    # 获取问题
    question = request.args.get('question')
    return analysis(question)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    try:
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y%m%d%H%M%S%f")
        file_name = time_string+'.pdf'
        # 将文件数据写入到磁盘上的文件中
        file = request.files['file']
        file.save(os.path.join(sys.path[0], 'files', file_name))
        persisit_database(file_name)
        return {'status': 'successful'}
    except Exception as e:
        print(e)
        return {'status': 'failure'}


def persisit_database(file_name):
    # 当前文件所在的目录
    global memorydb
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 打开PDF文件
    file_path = os.path.join(current_dir, 'files', file_name)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    embeddings = FakeEmbeddings(size=1352)
    persist_directory = os.path.join(current_dir, 'db')
    memorydb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory)
    memorydb.persist()


def analysis(question):
    global memorydb
    global chat_history
    try:
        if memorydb == None:
            embeddings = FakeEmbeddings(size=1352)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            persist_directory = os.path.join(current_dir, 'db')
            memorydb = Chroma(persist_directory=persist_directory,
                              embedding_function=embeddings)
            # print(asizeof.asizeof(memorydb))

        retriever = memorydb.as_retriever()
        model = ChatOpenAI(openai_api_base="http://127.0.0.1:1337",
                           openai_api_key="")
        qa = ConversationalRetrievalChain.from_llm(
            model, retriever=retriever, return_source_documents=True)

        result = qa({"question": question, "chat_history": chat_history})
        retain_5_conversation()
        chat_history.append((question, result["answer"]))
        arr_doc = []
        for ref_doc in result['source_documents']:
            arr_doc.append(
                {'page_content': ref_doc.page_content, 'page': ref_doc.metadata['page']})
        res_json = {
            'status': 200,
            'answer': result['answer'],
            'quote': arr_doc
        }
        return res_json
    except Exception as e:
        print(e)
        res_json = {
            'status': 500,
            'answer': 'server error'
        }
        return res_json


def retain_5_conversation():
    global chat_history
    if len(chat_history) >= 5:
        chat_history.pop(0)


if __name__ == '__main__':
    app.run(debug=True)
