import lazyllm

chat = lazyllm.OnlineChatModule('DeepSeek-V3',source="sensenova")
lazyllm.WebModule(chat, port=12367).start().wait()