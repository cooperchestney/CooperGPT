from CooperLLM import CooperLLM
import streamlit as st

@st.cache_data
def init_CooperLLM():
    llm = CooperLLM()
    return llm

CooperLLM = init_CooperLLM()

if 'personality' not in st.session_state:
    st.session_state['personality'] = 'helpful_bot'

if 'config' not in st.session_state:
    st.session_state['config'] = CooperLLM.default_config

if 'config_disabled' not in st.session_state:
    st.session_state['config_disabled'] = True

def update_config():
    for option in st.session_state['config']:
        st.session_state['config'][option] = st.session_state[option]
    CooperLLM.init_LLM(st.session_state['config'])
    CooperLLM.init_RAG(st.session_state['config'])
    st.session_state['config_disabled'] = True

def enable_config_button():
    st.session_state['config_disabled'] = False

st.title("CooperGPT")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What's up?")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    response = CooperLLM.chat(prompt, st.session_state['personality'], st.session_state['config'])
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

with st.sidebar:
    st.header('Current prompt')
    st.write(CooperLLM.prompt_options[st.session_state['personality']].split('\n\n', 1)[0])
    st.divider()
    st.subheader('Personality')
    st.selectbox(
        'Chatbot personality',
        options=CooperLLM.prompt_options,
        key='personality'
    )
    st.divider()
    with st.container():
        st.subheader('LLM configuration')
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state['config']['llm_temperature'],
            step=0.01,
            key='llm_temperature',
            on_change=enable_config_button       
        )
        st.number_input(
            "Max new tokens",
            min_value=1,
            max_value=400,
            value=st.session_state['config']['llm_max_new_tokens'],
            key='llm_max_new_tokens',
            on_change=enable_config_button  
        )
        st.number_input(
            "Splitter chunk size",
            min_value=1,
            max_value=500,
            value=st.session_state['config']['splitter_chunk_size'],
            key='splitter_chunk_size',
            on_change=enable_config_button  
        )
        st.number_input(
            "Splitter chunk overlap",
            min_value=0,
            max_value=50,
            value=st.session_state['config']['splitter_chunk_overlap'],
            key='splitter_chunk_overlap',
            on_change=enable_config_button  
        )
        st.number_input(
            "Retriever k",
            min_value=1,
            max_value=10,
            value=st.session_state['config']['retriever_k'],
            key = 'retriever_k',
            on_change=enable_config_button
        )
        st.button(
            "Update LLM configuration",
            on_click=update_config,
            disabled=st.session_state['config_disabled']
        )