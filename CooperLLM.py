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
        self.default_config = {'splitter_chunk_size': 200,
                       'splitter_chunk_overlap': 25,
                       'llm_max_new_tokens': 256,
                       'llm_temperature': 0.2,
                       'retriever_k': 3}
        self.llm_stop = ['\n\n']
        self.prompt_options = self.get_prompt_options()
        self.has_initialised = False
    
    def init_RAG(self, config):
        #initialise RAG variables
        self.loader = TextLoader(self.chat_log.CHAT_FILE_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=config['splitter_chunk_size'], 
                                                       chunk_overlap=config['splitter_chunk_overlap'])
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    def init_LLM(self, config):
        #initialise LLM 
        self.llm = CTransformers(model = 'openchat_3.5.Q6_K.gguf', 
                            model_type='mistral', 
                            config={'max_new_tokens': config['llm_max_new_tokens'], 
                                    'temperature': config['llm_temperature'], 
                                    'stop': self.llm_stop})
    
    def get_prompt_options(self):
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
        return prompt_options

    def select_prompt(self, prompt_option):       
        if prompt_option in self.prompt_options:
            return self.prompt_options[prompt_option]

    def vectorise_chat_log(self):
        #return chat log as a vector database
        chat_history = self.loader.load()
        chat_history = self.text_splitter.split_documents(chat_history)
        db = FAISS.from_documents(chat_history, self.embeddings)
        return db
    
    def get_LLM_with_chat_log(self, prompt_option, config):
        #get LLM with ability to access most updated chat log
        if self.has_initialised == False:
            self.has_initialised = True
            self.init_RAG(config)
            self.init_LLM(config)

        db = self.vectorise_chat_log()
        retriever = db.as_retriever(search_kwargs={'k': config['retriever_k']})

        prompt = PromptTemplate(
            template=self.select_prompt(prompt_option),
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

    def chat(self, request, prompt_option, config):
        qa_llm = self.get_LLM_with_chat_log(prompt_option, config)
        output = qa_llm({'query': request})
        response = output["result"]
        self.update_chat_log(request, response)
        return response






    
