from CooperLLM import CooperLLM
import streamlit as st

@st.cache_data
def init_CooperLLM():
    llm = CooperLLM()
    return llm

CooperLLM = init_CooperLLM()

def get_config(llm):
    personality_llm2app = {"helpful_bot": "Helpful", "mean_bot": "Mean"}

    llm_personality = llm.personality
    st.session_state["personality"] = personality_llm2app[personality]
    print(st.session_state["personality"])

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
    response = CooperLLM.chat(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

with st.sidebar:
    st.header('Current prompt')
    st.write(CooperLLM.config["prompt_template"].split('\n\n', 1)[0])
    personality = st.selectbox(
        'Which chatbot personality?',
        ('Helpful', 'Mean'),
        key='personality'
    )
    get_config(CooperLLM)

    st.write(personality)