import streamlit as st
import html
import re
from tavily import TavilyClient
from groq import Groq

# Bulut Güvenli API İstemcileri
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(page_title='Turgeo.AI Workspace', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 100% !important;
    }
    .user-row {
        display: flex;
        justify-content: flex-end;
        width: 100%;
        margin-bottom: 16px;
    }
    .user-bubble {
        background-color: #0078D7;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 16px 16px 2px 16px;
        max-width: 80%;
        word-wrap: break-word;
        word-break: break-word;
        white-space: pre-wrap;
        box-sizing: border-box;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }
    .ai-wrapper {
        margin-bottom: 24px;
        width: 100%;
        word-wrap: break-word;
        word-break: break-word;
    }
    .source-box {
        font-size: 13px;
        color: #a0a0a0;
        background: #1a1a1a;
        padding: 12px;
        border-radius: 6px;
        margin-top: 12px;
        border-left: 3px solid #0078D7;
    }
</style>
''', unsafe_allow_html=True)

def robust_web_search(query: str, max_results: int = 5):
    results = []
    display_links = []
    try:
        response = tavily.search(query=query, max_results=max_results, search_depth="advanced")
        for idx, r in enumerate(response.get('results', []), 1):
            title = r.get('title', 'Source')
            snippet = r.get('content', '')
            url = r.get('url', '#')
            
            results.append(f"[{idx}] Title: {title}\nSnippet: {snippet}")
            display_links.append(f"[{idx}] <a href='{url}' target='_blank' style='color:#4DA8DA; text-decoration: none;'>{title}</a>")
    except Exception as e:
        print(f"Tavily Search Error: {e}")
    return results, display_links

def analyze_search_need(user_prompt: str, chat_history: list) -> str:
    context_msgs = chat_history[-3:] if len(chat_history) >= 3 else chat_history
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in context_msgs])
    
    system_eval = (
        "You are an intelligent search query generator for a web search engine.\n"
        "Analyze the chat history and the latest user prompt.\n"
        "Rule 1: Output ONLY 'NONE' if the prompt is purely a greeting, small talk, general math calculation, or generic coding syntax that needs no external lookup.\n"
        "Rule 2: If a web search is needed, generate a complete, precise, and descriptive search query that preserves all key entities, locations, and context from the prompt.\n"
        "Do NOT output conversational filler, quotation marks, or explanations. Just output the search query text or NONE."
    )
    
    prompt_payload = f"Chat History:\n{history_str}\n\nLatest Prompt: {user_prompt}"
    
    try:
        res = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {'role': 'system', 'content': system_eval},
                {'role': 'user', 'content': prompt_payload}
            ],
            temperature=0.0,
            max_tokens=40
        )
        output = res.choices[0].message.content.strip()
        cleaned = output.replace('"', '').replace("'", "").strip()
        if 'NONE' in cleaned.upper() or len(cleaned) < 2:
            return "NONE"
        return cleaned
    except Exception:
        return user_prompt

with st.sidebar:
    st.markdown('### Turgeo.AI Core')
    web_search_enabled = st.toggle('Auto-Web Grounding', value=True)
    st.divider()
    if st.button('Clear Context', use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("<div style='margin-top: 20px; padding: 10px; font-size: 12px; color: #888;'>Engine: Llama 3.1 70B (Groq)<br>Cloud Deployed</div>", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

is_generating = len(st.session_state.messages) > 0 and st.session_state.messages[-1]['role'] == 'user'

if len(st.session_state.messages) == 0:
    st.markdown("<h1 style='text-align: center; color: #e3e3e3; margin-bottom: 10px;'>Turgeo.AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; margin-bottom: 30px;'>Cloud Autonomous Knowledge System</p>", unsafe_allow_html=True)

prompt = st.chat_input('Message Turgeo.AI...', disabled=is_generating)

for msg in st.session_state.messages:
    if msg['role'] == 'user':
        safe_text = html.escape(msg['content'])
        st.markdown(f'<div class="user-row"><div class="user-bubble">{safe_text}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ai-wrapper">', unsafe_allow_html=True)
        st.markdown(msg['content'])
        if 'sources' in msg and msg['sources']:
            st.markdown(f'<div class="source-box"><b>Verified Sources:</b><br>{msg["sources"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.rerun()

if is_generating:
    user_prompt = st.session_state.messages[-1]['content']
    search_context = ""
    sources_display_html = ""
    
    if web_search_enabled:
        with st.status("Evaluating and generating search query...", expanded=False) as status:
            search_keywords = analyze_search_need(user_prompt, st.session_state.messages[:-1])
            
            if search_keywords != "NONE":
                status.update(label=f"Searching web for: ''{search_keywords}''", state="running", expanded=False)
                try:
                    results, display_links = robust_web_search(search_keywords, max_results=5)
                            
                    if results:
                        search_context = "\n\n[REAL-TIME WEB DATA]:\n" + "\n---\n".join(results)
                        sources_display_html = "<br>".join(display_links)
                        status.update(label=f"Searched: ''{search_keywords}'' ({len(results)} results found)", state="complete", expanded=False)
                    else:
                        status.update(label=f"Searched ''{search_keywords}'', but returned 0 results.", state="complete", expanded=False)
                except Exception as e:
                    status.update(label=f"Search Error: {str(e)}", state="error", expanded=False)
            else:
                status.update(label="No web search needed. Responding directly.", state="complete", expanded=False)

    chat_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    
    if search_context:
        chat_messages[-1]['content'] += f"{search_context}\n\nINSTRUCTION: Analyze the above [REAL-TIME WEB DATA] thoroughly along with the conversation history. Synthesize a precise, accurate, and direct response."

    st.markdown('<div class="ai-wrapper">', unsafe_allow_html=True)
    response_placeholder = st.empty()
    full_response = ''
    
    try:
        stream = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=chat_messages,
            stream=True,
            temperature=0.2,
            max_tokens=4096,
            top_p=0.9
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                response_placeholder.markdown(full_response + " ▌")
            
        response_placeholder.markdown(full_response)
        
        if sources_display_html:
            st.markdown(f'<div class="source-box"><b>Verified Sources:</b><br>{sources_display_html}</div>', unsafe_allow_html=True)
            
    except Exception as e:
        full_response = f"Execution Error: {str(e)}"
        response_placeholder.markdown(full_response)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.session_state.messages.append({
        'role': 'assistant', 
        'content': full_response, 
        'sources': sources_display_html
    })
    st.rerun()
