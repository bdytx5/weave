"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[1260],{3128:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>r,default:()=>d,frontMatter:()=>s,metadata:()=>a,toc:()=>l});var o=n(85893),i=n(11151);const s={},r="Together AI",a={id:"guides/integrations/together_ai",title:"Together AI",description:"Together AI is a platform for building and finetuning generative AI models, focusing on Open Source LLMs, and allowing customers to fine-tune and host their own models.",source:"@site/docs/guides/integrations/together_ai.md",sourceDirName:"guides/integrations",slug:"/guides/integrations/together_ai",permalink:"/guides/integrations/together_ai",draft:!1,unlisted:!1,editUrl:"https://github.com/wandb/weave/blob/master/docs/docs/guides/integrations/together_ai.md",tags:[],version:"current",lastUpdatedAt:1727132343e3,frontMatter:{},sidebar:"documentationSidebar",previous:{title:"Google Gemini",permalink:"/guides/integrations/google-gemini"},next:{title:"Groq",permalink:"/guides/integrations/groq"}},c={},l=[];function h(e){const t={a:"a",admonition:"admonition",code:"code",h1:"h1",p:"p",pre:"pre",...(0,i.a)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(t.h1,{id:"together-ai",children:"Together AI"}),"\n",(0,o.jsx)(t.p,{children:"Together AI is a platform for building and finetuning generative AI models, focusing on Open Source LLMs, and allowing customers to fine-tune and host their own models."}),"\n",(0,o.jsx)(t.admonition,{type:"info",children:(0,o.jsxs)(t.p,{children:["Full Weave ",(0,o.jsx)(t.code,{children:"together"})," python package support is currently in development"]})}),"\n",(0,o.jsxs)(t.p,{children:["While full Weave support for the ",(0,o.jsx)(t.code,{children:"together"})," python package is currently in development, Together supports the OpenAI SDK compatibility (",(0,o.jsx)(t.a,{href:"https://docs.together.ai/docs/openai-api-compatibility",children:"docs"}),") which Weave automatically detects and integrates with."]}),"\n",(0,o.jsxs)(t.p,{children:["To switch to using the Together API, simply switch out the API key to your ",(0,o.jsx)(t.a,{href:"https://docs.together.ai/docs/get-started#access-your-api-key",children:"Together API"})," key, ",(0,o.jsx)(t.code,{children:"base_url"})," to ",(0,o.jsx)(t.code,{children:"https://api.together.xyz/v1"}),", and model to one of their ",(0,o.jsx)(t.a,{href:"https://docs.together.ai/docs/inference-models#chat-models",children:"chat models"}),"."]}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",children:'import os\nimport openai\nimport weave\n\n# highlight-next-line\nweave.init(\'together-weave\')\n\nsystem_content = "You are a travel agent. Be descriptive and helpful."\nuser_content = "Tell me about San Francisco"\n\n# highlight-next-line\nclient = openai.OpenAI(\n# highlight-next-line\n    api_key=os.environ.get("TOGETHER_API_KEY"),\n# highlight-next-line\n    base_url="https://api.together.xyz/v1",\n# highlight-next-line\n)\nchat_completion = client.chat.completions.create(\n    model="mistralai/Mixtral-8x7B-Instruct-v0.1",\n    messages=[\n        {"role": "system", "content": system_content},\n        {"role": "user", "content": user_content},\n    ],\n    temperature=0.7,\n    max_tokens=1024,\n)\nresponse = chat_completion.choices[0].message.content\nprint("Together response:\\n", response)\n'})}),"\n",(0,o.jsxs)(t.p,{children:["While this is a simple example to get started, see our ",(0,o.jsx)(t.a,{href:"/guides/integrations/openai#track-your-own-ops",children:"OpenAI"})," guide for more details on how to integrate Weave with your own functions for more complex use cases."]})]})}function d(e={}){const{wrapper:t}={...(0,i.a)(),...e.components};return t?(0,o.jsx)(t,{...e,children:(0,o.jsx)(h,{...e})}):h(e)}},11151:(e,t,n)=>{n.d(t,{Z:()=>a,a:()=>r});var o=n(67294);const i={},s=o.createContext(i);function r(e){const t=o.useContext(s);return o.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function a(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:r(e.components),o.createElement(s.Provider,{value:t},e.children)}}}]);