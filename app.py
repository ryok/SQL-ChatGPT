import sys
import os
import re
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.llms.openai import OpenAI
import gradio as gr
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor

# search = GoogleSearchAPIWrapper()
db = SQLDatabase.from_uri("sqlite:///./Chinook.db")


def on_token_change(user_token, state):
    print(user_token)
    openai.api_key = user_token or os.environ.get("OPENAI_API_KEY")
    state["user_token"] = user_token
    return state


class ConversationBot:
    def __init__(self):
        print("Initializing ChatGPT")
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.toolkit = SQLDatabaseToolkit(db=db)
        self.agent = create_sql_agent(
            toolkit=self.toolkit,
            llm=self.llm,
            agent="conversational-react-description",
            memory=self.memory,
            verbose=True)

    def run_text(self, text, state):
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        print("======>Current memory:\n %s" % self.agent.memory)
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state

        
if __name__ == '__main__':
    bot = ConversationBot()
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="DB integrated ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.8):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
            with gr.Column(scale=0.2, min_width=0):
                clear = gr.Button("Clear️")
        with gr.Row():
            with gr.Column():
                gr.Markdown("Enter your own OpenAI API Key to try out more than 5 times. You can get it [here](https://platform.openai.com/account/api-keys).")
                user_token = gr.Textbox(placeholder="OpenAI API Key", type="password", show_label=False)

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        user_token.change(on_token_change, inputs=[user_token, state], outputs=[state])
        demo.launch(server_name="0.0.0.0", server_port=8080)