from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class ChatLog:
    #functions to create and manage a text file of chat history
    def __init__(self):
        self.CHAT_FILE_NAME = "chat_history.txt"
        self.clear_chat_history()

    def clear_chat_history(self):
        with open(self.CHAT_FILE_NAME, 'w', encoding="utf-8") as text_file:
            text_file.write("Chat log:")
            
    def read_chat_history(self):
        with open(self.CHAT_FILE_NAME, 'r', encoding="utf-8") as text_file:
            return text_file.read()
        
    def append_chat_history(self, chat):
        prev_chat = self.read_chat_history()
        new_chat_history = prev_chat + chat
        with open(self.CHAT_FILE_NAME, 'w', encoding="utf-8") as text_file:
            text_file.write(new_chat_history)

class CooperLLM:
    def __init__(self):
        self.chat_log = ChatLog()
        self.config = {'splitter_chunk_size': 200,
                       'splitter_chunk_overlap': 25,
                       'llm_max_new_tokens': 256,
                       'llm_temperature': 0.2,
                       'llm_stop': ['\n\n'],
                       'retriever_k': 3,
                       'prompt_template': '''
                        You are a helpful chatbot that answers questions based on chat history and your own knowledge.
                        Only answer the request and nothing more. Try to be concise.

                        {context}

                        Request: {question}

                        Helpful response:
                       '''}
        self.personality = 'helpful_bot'
        self.choose_prompt('mean_bot')
        self.has_initialised = False
    
    def init_RAG(self):
        #initialise RAG variables
        self.loader = TextLoader(self.chat_log.CHAT_FILE_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config['splitter_chunk_size'], 
                                                       chunk_overlap=self.config['splitter_chunk_overlap'])
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    def init_LLM(self):
        #initialise LLM 
        self.llm = CTransformers(model = 'openchat_3.5.Q6_K.gguf', 
                            model_type='mistral', 
                            config={'max_new_tokens': self.config['llm_max_new_tokens'], 
                                    'temperature': self.config['llm_temperature'], 
                                    'stop': self.config['llm_stop']})
        
    def update_settings(self, new_configs):
        for key in self.config.keys():
            if key in new_configs:
                self.config[key] = new_configs[key]
    
    def choose_prompt(self, prompt_option):
        prompt_options = {
            'helpful_bot':
            '''
            You are a helpful chatbot that answers questions based on chat history and your own knowledge.
            Only answer the request and nothing more. Try to be concise.

            {context}

            Request: {question}

            Helpful response:
            ''',
            'mean_bot':
            '''
            You are a rude and unhelpful chatbot that answers questions based on chat history and your own knowledge.
            Only answer the request and nothing more. Try to be concise.

            {context}

            Request: {question}

            Rude response:
            '''}
        
        if prompt_option in prompt_options:
            self.personality = prompt_option
            self.config['prompt_template'] = prompt_options[prompt_option]

    def vectorise_chat_log(self):
        #return chat log as a vector database
        chat_history = self.loader.load()
        chat_history = self.text_splitter.split_documents(chat_history)
        db = FAISS.from_documents(chat_history, self.embeddings)
        return db
    
    def get_LLM_with_chat_log(self):
        #get LLM with ability to access most updated chat log
        if self.has_initialised == False:
            self.has_initialised = True
            self.init_RAG()
            self.init_LLM()

        db = self.vectorise_chat_log()
        retriever = db.as_retriever(search_kwargs={'k': self.config['retriever_k']})

        prompt = PromptTemplate(
            template=self.config['prompt_template'],
            input_variables=['context', 'question'])
        
        qa_llm = RetrievalQA.from_chain_type(llm=self.llm,
                                    chain_type='stuff',
                                    retriever=retriever,
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt': prompt}) 
        return qa_llm
    
    def update_chat_log(self, request, response):
        chat_update = '\n\nRequest: ' + request + '\nResponse: ' + response
        self.chat_log.append_chat_history(chat_update)

    def chat(self, request):
        qa_llm = self.get_LLM_with_chat_log()
        output = qa_llm({'query': request})
        response = output["result"]
        self.update_chat_log(request, response)
        return response






    
