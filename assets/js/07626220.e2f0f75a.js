"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2196],{48779:(e,n,a)=>{a.r(n),a.d(n,{assets:()=>S,contentTitle:()=>A,default:()=>q,frontMatter:()=>N,metadata:()=>P,toc:()=>L});var t=a(85893),l=a(11151),i=a(67294),s=a(90512),r=a(12466),c=a(70989),d=a(72389);const o={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};function h(e){let{className:n,block:a,selectedValue:l,selectValue:i,tabValues:c}=e;const d=[],{blockElementScrollPositionUntilNextRender:h}=(0,r.o5)(),u=e=>{const n=e.currentTarget,a=d.indexOf(n),t=c[a].value;t!==l&&(h(n),i(t))},p=e=>{let n=null;switch(e.key){case"Enter":u(e);break;case"ArrowRight":{const a=d.indexOf(e.currentTarget)+1;n=d[a]??d[0];break}case"ArrowLeft":{const a=d.indexOf(e.currentTarget)-1;n=d[a]??d[d.length-1];break}}n?.focus()};return(0,t.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,s.Z)("tabs",{"tabs--block":a},n),children:c.map((e=>{let{value:n,label:a,attributes:i}=e;return(0,t.jsx)("li",{role:"tab",tabIndex:l===n?0:-1,"aria-selected":l===n,ref:e=>d.push(e),onKeyDown:p,onClick:u,...i,className:(0,s.Z)("tabs__item",o.tabItem,i?.className,{"tabs__item--active":l===n}),children:a??n},n)}))})}function u(e){let{lazy:n,children:a,selectedValue:l}=e;const s=(Array.isArray(a)?a:[a]).filter(Boolean);if(n){const e=s.find((e=>e.props.value===l));return e?(0,i.cloneElement)(e,{className:"margin-top--md"}):null}return(0,t.jsx)("div",{className:"margin-top--md",children:s.map(((e,n)=>(0,i.cloneElement)(e,{key:n,hidden:e.props.value!==l})))})}function p(e){const n=(0,c.Y)(e);return(0,t.jsxs)("div",{className:(0,s.Z)("tabs-container",o.tabList),children:[(0,t.jsx)(h,{...n,...e}),(0,t.jsx)(u,{...n,...e})]})}function m(e){const n=(0,d.default)();return(0,t.jsx)(p,{...e,children:(0,c.h)(e.children)},String(n))}var x=a(85162);const j="window-content_wMkM",f="carousel_ivBY",g="carousel-inner_Kp_Z",y="carousel-item_ZkMm",v="carousel-image_BeQl",b="carousel-footer_PmSO",w="indicator-dot_joJu",_="active_bLKf",k=e=>{let{images:n,alt:a}=e;const[l,r]=(0,i.useState)(0),c=(0,i.useRef)(null),d=()=>{if(c.current){const e=c.current.scrollLeft,n=c.current.offsetWidth,a=Math.round(e/n);r(a)}};(0,i.useEffect)((()=>{const e=c.current;if(e)return e.addEventListener("scroll",d),()=>e.removeEventListener("scroll",d)}),[]);return Array.isArray(n)&&0!==n.length?(0,t.jsxs)("div",{className:j,children:[(0,t.jsx)("div",{ref:c,className:f,children:(0,t.jsx)("div",{className:g,children:n.map(((e,n)=>(0,t.jsx)("div",{className:y,children:(0,t.jsx)("img",{src:e,alt:`${a} ${n+1}`,className:v})},n)))})}),n.length>1&&(0,t.jsx)("div",{className:b,children:n.map(((e,n)=>(0,t.jsx)("button",{className:(0,s.Z)(w,n===l&&_),onClick:()=>(e=>{if(c.current){const n=c.current.offsetWidth;c.current.scrollTo({left:e*n,behavior:"smooth"})}})(n),"aria-label":`Go to image ${n+1}`},n)))})]}):null},C=a.p+"assets/images/calls_macro-8ffdcc1a6e206f442ce22245d0278c78.png",T=a.p+"assets/images/calls_filter-ad0e853eae355179a98716a8ddd34c6a.png",I=a.p+"assets/images/basic_call-d333612aef700095ccc45371891486e8.png",N={},A="Calls",P={id:"guides/tracking/tracing",title:"Calls",description:"<DesktopWindow",source:"@site/docs/guides/tracking/tracing.mdx",sourceDirName:"guides/tracking",slug:"/guides/tracking/tracing",permalink:"/guides/tracking/tracing",draft:!1,unlisted:!1,editUrl:"https://github.com/wandb/weave/blob/master/docs/docs/guides/tracking/tracing.mdx",tags:[],version:"current",lastUpdatedAt:1727132343e3,frontMatter:{},sidebar:"documentationSidebar",previous:{title:"Tracing",permalink:"/guides/tracking/"},next:{title:"Ops",permalink:"/guides/tracking/ops"}},S={},L=[{value:"Creating Calls",id:"creating-calls",level:2},{value:"1. Automatic Tracking of LLM Libraries",id:"1-automatic-tracking-of-llm-libraries",level:3},{value:"2. Decorating Functions",id:"2-decorating-functions",level:3},{value:"Call Display Name",id:"call-display-name",level:4},{value:"Attributes",id:"attributes",level:4},{value:"3. Manual Call Tracking",id:"3-manual-call-tracking",level:3},{value:"Viewing Calls",id:"viewing-calls",level:2},{value:"Updating Calls",id:"updating-calls",level:2},{value:"Set Display Name",id:"set-display-name",level:3},{value:"Add Feedback",id:"add-feedback",level:3},{value:"Delete a Call",id:"delete-a-call",level:3},{value:"Querying &amp; Exporting Calls",id:"querying--exporting-calls",level:2},{value:"Call FAQ",id:"call-faq",level:2},{value:"Call Schema",id:"call-schema",level:4}];function D(e){const n={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",h3:"h3",h4:"h4",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",ul:"ul",...(0,l.a)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h1,{id:"calls",children:"Calls"}),"\n",(0,t.jsx)(k,{images:[C,I,T],alt:"Screenshot of Weave Calls",title:"Weave Calls"}),"\n",(0,t.jsxs)(n.admonition,{title:"Calls",type:"info",children:[(0,t.jsx)(n.p,{children:"Calls are the fundamental building block in Weave. They represent a single execution of a function, including:"}),(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Inputs (arguments)"}),"\n",(0,t.jsx)(n.li,{children:"Outputs (return value)"}),"\n",(0,t.jsx)(n.li,{children:"Metadata (duration, exceptions, LLM usage, etc.)"}),"\n"]}),(0,t.jsxs)(n.p,{children:["Calls are similar to spans in the ",(0,t.jsx)(n.a,{href:"https://opentelemetry.io",children:"OpenTelemetry"})," data model. A Call can:"]}),(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Belong to a Trace (a collection of calls in the same execution context)"}),"\n",(0,t.jsx)(n.li,{children:"Have parent and child Calls, forming a tree structure"}),"\n"]})]}),"\n",(0,t.jsx)(n.h2,{id:"creating-calls",children:"Creating Calls"}),"\n",(0,t.jsx)(n.p,{children:"There are three main ways to create Calls in Weave:"}),"\n",(0,t.jsx)(n.h3,{id:"1-automatic-tracking-of-llm-libraries",children:"1. Automatic Tracking of LLM Libraries"}),"\n",(0,t.jsxs)(n.p,{children:["Weave automatically tracks ",(0,t.jsx)(n.a,{href:"/guides/integrations/",children:"calls to common LLM libraries"})," like ",(0,t.jsx)(n.code,{children:"openai"}),", ",(0,t.jsx)(n.code,{children:"anthropic"}),", ",(0,t.jsx)(n.code,{children:"cohere"}),", and ",(0,t.jsx)(n.code,{children:"mistral"}),". Simply call ",(0,t.jsx)(n.a,{href:"/reference/python-sdk/weave/#function-init",children:(0,t.jsx)(n.code,{children:"weave.init('project_name')"})})," at the start of your program:"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'import weave\n\nfrom openai import OpenAI\nclient = OpenAI()\n\n# Initialize Weave Tracing\nweave.init(\'intro-example\')\n\nresponse = client.chat.completions.create(\n  model="gpt-4",\n  messages=[\n    {\n      "role": "user",\n      "content": "How are you?"\n    }\n  ],\n  temperature=0.8,\n  max_tokens=64,\n  top_p=1\n)\n'})}),"\n",(0,t.jsx)(n.h3,{id:"2-decorating-functions",children:"2. Decorating Functions"}),"\n",(0,t.jsxs)(n.p,{children:["However, often LLM applications have additional logic (such as pre/post processing, prompts, etc.) that you want to track.\nWeave allows you to manually track these calls using the ",(0,t.jsx)(n.a,{href:"/reference/python-sdk/weave/#function-op",children:(0,t.jsx)(n.code,{children:"@weave.op"})})," decorator. For example:"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'import weave\n\n# Initialize Weave Tracing\nweave.init(\'intro-example\')\n\n# Decorate your function\n@weave.op\ndef my_function(name: str):\n    return f"Hello, {name}!"\n\n# Call your function -- Weave will automatically track inputs and outputs\nprint(my_function.call("World"))\n'})}),"\n",(0,t.jsx)(n.admonition,{type:"note",children:(0,t.jsx)(n.p,{children:"If your op is a method on a class, you need to pass the instance as the first argument to the op (see example below)."})}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'import weave\n\n# Initialize Weave Tracing\nweave.init("intro-example")\n\nclass MyClass:\n    # Decorate your method\n    @weave.op\n    def my_method(self, name: str):\n        return f"Hello, {name}!"\n\ninstance = MyClass()\n\n# Call your method -- Weave will automatically track inputs and outputs\ninstance.my_method.call(instance, "World")\n'})}),"\n",(0,t.jsx)(n.h4,{id:"call-display-name",children:"Call Display Name"}),"\n",(0,t.jsx)(n.p,{children:"Sometimes you may want to override the display name of a call. You can achieve this in one of three ways:"}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["Change the display name on a per-call basis. This uses the ",(0,t.jsx)(n.a,{href:"/reference/python-sdk/weave/trace/weave.trace.op#function-call",children:(0,t.jsx)(n.code,{children:"Op.call"})})," method to return a ",(0,t.jsx)(n.code,{children:"Call"})," object, which you can then use to set the display name using ",(0,t.jsx)(n.a,{href:"/reference/python-sdk/weave/trace/weave.trace.weave_client#method-set_display_name",children:(0,t.jsx)(n.code,{children:"Call.set_display_name"})}),"."]}),"\n"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'result, call = my_function.call("World")\ncall.set_display_name("My Custom Display Name")\n'})}),"\n",(0,t.jsxs)(n.ol,{start:"2",children:["\n",(0,t.jsx)(n.li,{children:"Change the display name for all Calls of a given Op:"}),"\n"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'@weave.op(call_display_name="My Custom Display Name")\ndef my_function(name: str):\n    return f"Hello, {name}!"\n'})}),"\n",(0,t.jsxs)(n.ol,{start:"3",children:["\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsxs)(n.p,{children:["The ",(0,t.jsx)(n.code,{children:"call_display_name"})," can also be a function that takes in a ",(0,t.jsx)(n.code,{children:"Call"})," object and returns a string.  The ",(0,t.jsx)(n.code,{children:"Call"})," object will be passed automatically when the function is called, so you can use it to dynamically generate names based on the function's name, call inputs, attributes, etc."]}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsx)(n.p,{children:"One common use case is just appending a timestamp to the function's name."}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-py",children:'from datetime import datetime\n\n@weave.op(call_display_name=lambda call: f"{call.func_name}__{datetime.now()}")\ndef func():\n    return ...\n'})}),"\n"]}),"\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsx)(n.p,{children:"You can also get creative with custom attributes"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-py",children:'def custom_attribute_name(call):\n    model = call.attributes["model"]\n    revision = call.attributes["revision"]\n    now = call.attributes["date"]\n\n    return f"{model}__{revision}__{now}"\n\n@weave.op(call_display_name=custom_attribute_name)\ndef func():\n    return ...\n\nwith weave.attributes(\n    {\n        "model": "finetuned-llama-3.1-8b",\n        "revision": "v0.1.2",\n        "date": "2024-08-01",\n    }\n):\n    func()  # the display name will be "finetuned-llama-3.1-8b__v0.1.2__2024-08-01"\n\n\n    with weave.attributes(\n        {\n            "model": "finetuned-gpt-4o",\n            "revision": "v0.1.3",\n            "date": "2024-08-02",\n        }\n    ):\n        func()  # the display name will be "finetuned-gpt-4o__v0.1.3__2024-08-02"\n'})}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(n.p,{children:[(0,t.jsx)(n.strong,{children:"Technical Note:"}),' "Calls" are produced by "Ops". An Op is a function or method that is decorated with ',(0,t.jsx)(n.code,{children:"@weave.op"}),".\nBy default, the Op's name is the function name, and the associated calls will have the same display name. The above example shows how to override the display name for all Calls of a given Op.  Sometimes, users wish to override the name of the Op itself. This can be achieved in one of two ways:"]}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["Set the ",(0,t.jsx)(n.code,{children:"name"})," property of the Op before any calls are logged"]}),"\n"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'my_function.name = "My Custom Op Name"\n'})}),"\n",(0,t.jsxs)(n.ol,{start:"2",children:["\n",(0,t.jsxs)(n.li,{children:["Set the ",(0,t.jsx)(n.code,{children:"name"})," option on the op decorator"]}),"\n"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'@weave.op(name="My Custom Op Name)\n'})}),"\n",(0,t.jsx)(n.h4,{id:"attributes",children:"Attributes"}),"\n",(0,t.jsxs)(n.p,{children:["When calling tracked functions, you can add additional metadata to the call by using the ",(0,t.jsx)(n.a,{href:"/reference/python-sdk/weave/#function-attributes",children:(0,t.jsx)(n.code,{children:"weave.attributes"})})," context manager. In the example below, we add an ",(0,t.jsx)(n.code,{children:"env"})," attribute to the call specified as ",(0,t.jsx)(n.code,{children:"'production'"}),"."]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:"# ... continued from above ...\n\n# Add additional \"attributes\" to the call\nwith weave.attributes({'env': 'production'}):\n    print(my_function.call(\"World\"))\n"})}),"\n",(0,t.jsx)(n.h3,{id:"3-manual-call-tracking",children:"3. Manual Call Tracking"}),"\n",(0,t.jsx)(n.p,{children:"You can also manually create Calls using the API directly."}),"\n",(0,t.jsxs)(m,{groupId:"client-layer",children:[(0,t.jsx)(x.default,{value:"python_sdk",label:"Python",children:(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'import weave\n\n# Initialize Weave Tracing\nweave.init(\'intro-example\')\n\ndef my_function(name: str):\n    # Start a call\n    call = weave.create_call(op="my_function", inputs={"name": name})\n\n    # ... your function code ...\n\n    # End a call\n    weave.finish_call(call, output="Hello, World!")\n\n# Call your function\nprint(my_function("World"))\n'})})}),(0,t.jsxs)(x.default,{value:"service_api",label:"HTTP API",children:[(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["Start a call: ",(0,t.jsxs)(n.a,{href:"/reference/service-api/call-start-call-start-post",children:["POST ",(0,t.jsx)(n.code,{children:"/call/start"})]})]}),"\n",(0,t.jsxs)(n.li,{children:["End a call: ",(0,t.jsxs)(n.a,{href:"/reference/service-api/call-end-call-end-post",children:["POST ",(0,t.jsx)(n.code,{children:"/call/end"})]})]}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:'curl -L \'https://trace.wandb.ai/call/start\' \\\n-H \'Content-Type: application/json\' \\\n-H \'Accept: application/json\' \\\n-d \'{\n    "start": {\n        "project_id": "string",\n        "id": "string",\n        "op_name": "string",\n        "display_name": "string",\n        "trace_id": "string",\n        "parent_id": "string",\n        "started_at": "2024-09-08T20:07:34.849Z",\n        "attributes": {},\n        "inputs": {},\n        "wb_run_id": "string"\n    }\n}\n'})})]})]}),"\n",(0,t.jsx)(n.h2,{id:"viewing-calls",children:"Viewing Calls"}),"\n",(0,t.jsxs)(m,{groupId:"client-layer",children:[(0,t.jsxs)(x.default,{value:"web_app",label:"Web App",children:[(0,t.jsx)(n.p,{children:"To view a call in the web app:"}),(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:'Navigate to your project\'s "Traces" tab'}),"\n",(0,t.jsx)(n.li,{children:"Find the call you want to view in the list"}),"\n",(0,t.jsx)(n.li,{children:"Click on the call to open its details page"}),"\n"]}),(0,t.jsx)(n.p,{children:"The details page will show the call's inputs, outputs, runtime, and any additional attributes or metadata."}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.img,{alt:"View Call in Web App",src:a(61276).Z+"",width:"1920",height:"1080"})})]}),(0,t.jsxs)(x.default,{value:"python_sdk",label:"Python",children:[(0,t.jsxs)(n.p,{children:["To view a call using the Python API, you can use the ",(0,t.jsx)(n.a,{href:"../../reference/python-sdk/weave/trace/weave.trace.weave_client#method-get_call",children:(0,t.jsx)(n.code,{children:"get_call"})})," method:"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'import weave\n\n# Initialize the client\nclient = weave.init("your-project-name")\n\n# Get a specific call by its ID\ncall = client.get_call("call-uuid-here")\n\nprint(call)\n'})})]}),(0,t.jsxs)(x.default,{value:"service_api",label:"HTTP API",children:[(0,t.jsxs)(n.p,{children:["To view a call using the Service API, you can make a request to the ",(0,t.jsx)(n.a,{href:"/reference/service-api/call-read-call-read-post",children:(0,t.jsx)(n.code,{children:"/call/read"})})," endpoint."]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:"curl -L 'https://trace.wandb.ai/call/read' \\\n-H 'Content-Type: application/json' \\\n-H 'Accept: application/json' \\\n-d '{\n    \"project_id\": \"string\",\n    \"id\": \"string\",\n}'\n"})})]})]}),"\n",(0,t.jsx)(n.h2,{id:"updating-calls",children:"Updating Calls"}),"\n",(0,t.jsx)(n.p,{children:"Calls are mostly immutable once created, however, there are a few mutations which are supported:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:(0,t.jsx)(n.a,{href:"#set-display-name",children:"Set Display Name"})}),"\n",(0,t.jsx)(n.li,{children:(0,t.jsx)(n.a,{href:"#add-feedback",children:"Add Feedback"})}),"\n",(0,t.jsx)(n.li,{children:(0,t.jsx)(n.a,{href:"#delete-a-call",children:"Delete a Call"})}),"\n"]}),"\n",(0,t.jsx)(n.p,{children:"All of these mutations can be performed from the UI by navigating to the call detail page:"}),"\n",(0,t.jsx)(n.p,{children:(0,t.jsx)(n.img,{alt:"Update Call in Web App",src:a(42658).Z+"",width:"2078",height:"1124"})}),"\n",(0,t.jsx)(n.h3,{id:"set-display-name",children:"Set Display Name"}),"\n",(0,t.jsxs)(m,{groupId:"client-layer",children:[(0,t.jsxs)(x.default,{value:"python_sdk",label:"Python",children:[(0,t.jsxs)(n.p,{children:["In order to set the display name of a call, you can use the ",(0,t.jsx)(n.a,{href:"/reference/python-sdk/weave/trace/weave.trace.weave_client#method-set_display_name",children:(0,t.jsx)(n.code,{children:"Call.set_display_name"})})," method."]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'import weave\n\n# Initialize the client\nclient = weave.init("your-project-name")\n\n# Get a specific call by its ID\ncall = client.get_call("call-uuid-here")\n\n# Set the display name of the call\ncall.set_display_name("My Custom Display Name")\n'})})]}),(0,t.jsxs)(x.default,{value:"service_api",label:"HTTP API",children:[(0,t.jsxs)(n.p,{children:["To set the display name of a call using the Service API, you can make a request to the ",(0,t.jsx)(n.a,{href:"/reference/service-api/call-update-call-update-post",children:(0,t.jsx)(n.code,{children:"/call/update"})})," endpoint."]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:'curl -L \'https://trace.wandb.ai/call/update\' \\\n-H \'Content-Type: application/json\' \\\n-H \'Accept: application/json\' \\\n-d \'{\n    "project_id": "string",\n    "call_id": "string",\n    "display_name": "string",\n}\'\n'})})]})]}),"\n",(0,t.jsx)(n.h3,{id:"add-feedback",children:"Add Feedback"}),"\n",(0,t.jsxs)(n.p,{children:["Please see the ",(0,t.jsx)(n.a,{href:"/guides/tracking/feedback",children:"Feedback Documentation"})," for more details."]}),"\n",(0,t.jsx)(n.h3,{id:"delete-a-call",children:"Delete a Call"}),"\n",(0,t.jsxs)(m,{groupId:"client-layer",children:[(0,t.jsxs)(x.default,{value:"python_sdk",label:"Python",children:[(0,t.jsxs)(n.p,{children:["To delete a call using the Python API, you can use the ",(0,t.jsx)(n.a,{href:"/reference/python-sdk/weave/trace/weave.trace.weave_client#method-delete",children:(0,t.jsx)(n.code,{children:"Call.delete"})})," method."]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",metastring:"showLineNumbers",children:'import weave\n\n# Initialize the client\nclient = weave.init("your-project-name")\n\n# Get a specific call by its ID\ncall = client.get_call("call-uuid-here")\n\n# Delete the call\ncall.delete()\n'})})]}),(0,t.jsxs)(x.default,{value:"service_api",label:"HTTP API",children:[(0,t.jsxs)(n.p,{children:["To delete a call using the Service API, you can make a request to the ",(0,t.jsx)(n.a,{href:"/reference/service-api/calls-delete-calls-delete-post",children:(0,t.jsx)(n.code,{children:"/calls/delete"})})," endpoint."]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:"curl -L 'https://trace.wandb.ai/calls/delete' \\\n-H 'Content-Type: application/json' \\\n-H 'Accept: application/json' \\\n-d '{\n    \"project_id\": \"string\",\n    \"call_ids\": [\n        \"string\"\n    ],\n}'\n"})})]})]}),"\n",(0,t.jsx)(n.h2,{id:"querying--exporting-calls",children:"Querying & Exporting Calls"}),"\n",(0,t.jsx)(k,{images:[T],alt:"Screenshot of many calls",title:"Weave Calls"}),"\n",(0,t.jsxs)(n.p,{children:["The ",(0,t.jsx)(n.code,{children:"/calls"}),' page of your project ("Traces" tab) contains a table view of all the Calls in your project. From there, you can:']}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Sort"}),"\n",(0,t.jsx)(n.li,{children:"Filter"}),"\n",(0,t.jsx)(n.li,{children:"Export"}),"\n"]}),"\n",(0,t.jsx)(n.p,{children:(0,t.jsx)(n.img,{alt:"Calls Table View",src:a(64850).Z+"",width:"1704",height:"1564"})}),"\n",(0,t.jsx)(n.p,{children:"The Export Modal (shown above) allows you to export your data in a number of formats, as well as shows the Python & CURL equivalents for the selected calls!\nThe easiest way to get started is to construct a view in the UI, then learn more about the export API via the generated code snippets."}),"\n",(0,t.jsxs)(m,{groupId:"client-layer",children:[(0,t.jsxs)(x.default,{value:"python_sdk",label:"Python",children:[(0,t.jsxs)(n.p,{children:["To fetch calls using the Python API, you can use the ",(0,t.jsx)(n.a,{href:"/reference/python-sdk/weave/trace/weave.trace.weave_client#method-calls",children:(0,t.jsx)(n.code,{children:"client.calls"})})," method:"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'import weave\n\n# Initialize the client\nclient = weave.init("your-project-name")\n\n# Fetch calls\ncalls = client.calls(filter=...)\n'})}),(0,t.jsxs)(n.admonition,{title:"Notice: Evolving APIs",type:"info",children:[(0,t.jsxs)(n.p,{children:["Currently, it is easier to use the lower-level ",(0,t.jsx)(n.a,{href:"../../reference/python-sdk/weave/trace_server_bindings/weave.trace_server_bindings.remote_http_trace_server#method-calls_query_stream",children:(0,t.jsx)(n.code,{children:"calls_query_stream"})})," API as it is more flexible and powerful.\nIn the near future, we will move all functionality to the above client API."]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'import weave\n\n# Initialize the client\nclient = weave.init("your-project-name")\n\ncalls = client.server.calls_query_stream({\n    "project_id": "",\n    "filter": {},\n    "query": {},\n    "sort_by": [],\n})\n'})})]})]}),(0,t.jsxs)(x.default,{value:"service_api",label:"HTTP API",children:[(0,t.jsxs)(n.p,{children:["The most powerful query layer is at the Service API. To fetch calls using the Service API, you can make a request to the ",(0,t.jsx)(n.a,{href:"/reference/service-api/calls-query-stream-calls-stream-query-post",children:(0,t.jsx)(n.code,{children:"/calls/stream_query"})})," endpoint."]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:'curl -L \'https://trace.wandb.ai/calls/stream_query\' \\\n-H \'Content-Type: application/json\' \\\n-H \'Accept: application/json\' \\\n-d \'{\n"project_id": "string",\n"filter": {\n    "op_names": [\n        "string"\n    ],\n    "input_refs": [\n        "string"\n    ],\n    "output_refs": [\n        "string"\n    ],\n    "parent_ids": [\n        "string"\n    ],\n    "trace_ids": [\n        "string"\n    ],\n    "call_ids": [\n        "string"\n    ],\n    "trace_roots_only": true,\n    "wb_user_ids": [\n        "string"\n    ],\n    "wb_run_ids": [\n        "string"\n    ]\n},\n"limit": 100,\n"offset": 0,\n"sort_by": [\n    {\n    "field": "string",\n    "direction": "asc"\n    }\n],\n"query": {\n    "$expr": {}\n},\n"include_costs": true,\n"include_feedback": true,\n"columns": [\n    "string"\n],\n"expand_columns": [\n    "string"\n]\n}\'\n'})})]})]}),"\n","\n",(0,t.jsx)(n.h2,{id:"call-faq",children:"Call FAQ"}),"\n","\n",(0,t.jsx)(n.h4,{id:"call-schema",children:"Call Schema"}),"\n",(0,t.jsxs)(n.p,{children:["Please see the ",(0,t.jsx)(n.a,{href:"../../reference/python-sdk/weave/trace_server/weave.trace_server.trace_server_interface#class-callschema",children:"schema"})," for a complete list of fields."]}),"\n",(0,t.jsxs)(n.table,{children:[(0,t.jsx)(n.thead,{children:(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.th,{children:"Property"}),(0,t.jsx)(n.th,{children:"Type"}),(0,t.jsx)(n.th,{children:"Description"})]})}),(0,t.jsxs)(n.tbody,{children:[(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"id"}),(0,t.jsx)(n.td,{children:"string (uuid)"}),(0,t.jsx)(n.td,{children:"Unique identifier for the call"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"project_id"}),(0,t.jsx)(n.td,{children:"string (optional)"}),(0,t.jsx)(n.td,{children:"Associated project identifier"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"op_name"}),(0,t.jsx)(n.td,{children:"string"}),(0,t.jsx)(n.td,{children:"Name of the operation (can be a reference)"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"display_name"}),(0,t.jsx)(n.td,{children:"string (optional)"}),(0,t.jsx)(n.td,{children:"User-friendly name for the call"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"trace_id"}),(0,t.jsx)(n.td,{children:"string (uuid)"}),(0,t.jsx)(n.td,{children:"Identifier for the trace this call belongs to"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"parent_id"}),(0,t.jsx)(n.td,{children:"string (uuid)"}),(0,t.jsx)(n.td,{children:"Identifier of the parent call"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"started_at"}),(0,t.jsx)(n.td,{children:"datetime"}),(0,t.jsx)(n.td,{children:"Timestamp when the call started"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"attributes"}),(0,t.jsx)(n.td,{children:"Dict[str, Any]"}),(0,t.jsx)(n.td,{children:"User-defined metadata about the call"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"inputs"}),(0,t.jsx)(n.td,{children:"Dict[str, Any]"}),(0,t.jsx)(n.td,{children:"Input parameters for the call"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"ended_at"}),(0,t.jsx)(n.td,{children:"datetime (optional)"}),(0,t.jsx)(n.td,{children:"Timestamp when the call ended"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"exception"}),(0,t.jsx)(n.td,{children:"string (optional)"}),(0,t.jsx)(n.td,{children:"Error message if the call failed"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"output"}),(0,t.jsx)(n.td,{children:"Any (optional)"}),(0,t.jsx)(n.td,{children:"Result of the call"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"summary"}),(0,t.jsx)(n.td,{children:"Optional[SummaryMap]"}),(0,t.jsx)(n.td,{children:"Post-execution summary information"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"wb_user_id"}),(0,t.jsx)(n.td,{children:"Optional[str]"}),(0,t.jsx)(n.td,{children:"Associated Weights & Biases user ID"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"wb_run_id"}),(0,t.jsx)(n.td,{children:"Optional[str]"}),(0,t.jsx)(n.td,{children:"Associated Weights & Biases run ID"})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:"deleted_at"}),(0,t.jsx)(n.td,{children:"datetime (optional)"}),(0,t.jsx)(n.td,{children:"Timestamp of call deletion, if applicable"})]})]})]}),"\n",(0,t.jsx)(n.p,{children:"The table above outlines the key properties of a Call in Weave. Each property plays a crucial role in tracking and managing function calls:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["The ",(0,t.jsx)(n.code,{children:"id"}),", ",(0,t.jsx)(n.code,{children:"trace_id"}),", and ",(0,t.jsx)(n.code,{children:"parent_id"})," fields help in organizing and relating calls within the system."]}),"\n",(0,t.jsxs)(n.li,{children:["Timing information (",(0,t.jsx)(n.code,{children:"started_at"}),", ",(0,t.jsx)(n.code,{children:"ended_at"}),") allows for performance analysis."]}),"\n",(0,t.jsxs)(n.li,{children:["The ",(0,t.jsx)(n.code,{children:"attributes"})," and ",(0,t.jsx)(n.code,{children:"inputs"})," fields provide context for the call, while ",(0,t.jsx)(n.code,{children:"output"})," and ",(0,t.jsx)(n.code,{children:"summary"})," capture the results."]}),"\n",(0,t.jsxs)(n.li,{children:["Integration with Weights & Biases is facilitated through ",(0,t.jsx)(n.code,{children:"wb_user_id"})," and ",(0,t.jsx)(n.code,{children:"wb_run_id"}),"."]}),"\n"]}),"\n",(0,t.jsx)(n.p,{children:"This comprehensive set of properties enables detailed tracking and analysis of function calls throughout your project."}),"\n",(0,t.jsx)(n.p,{children:"Calculated Fields:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Cost"}),"\n",(0,t.jsx)(n.li,{children:"Duration"}),"\n",(0,t.jsx)(n.li,{children:"Status"}),"\n"]})]})}function q(e={}){const{wrapper:n}={...(0,l.a)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(D,{...e})}):D(e)}},85162:(e,n,a)=>{a.r(n),a.d(n,{default:()=>s});a(67294);var t=a(90512);const l={tabItem:"tabItem_Ymn6"};var i=a(85893);function s(e){let{children:n,hidden:a,className:s}=e;return(0,i.jsx)("div",{role:"tabpanel",className:(0,t.Z)(l.tabItem,s),hidden:a,children:n})}},70989:(e,n,a)=>{a.d(n,{Y:()=>p,h:()=>d});var t=a(67294),l=a(16550),i=a(20469),s=a(91980),r=a(67392),c=a(20812);function d(e){return t.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,t.isValidElement)(e)&&function(e){const{props:n}=e;return!!n&&"object"==typeof n&&"value"in n}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function o(e){const{values:n,children:a}=e;return(0,t.useMemo)((()=>{const e=n??function(e){return d(e).map((e=>{let{props:{value:n,label:a,attributes:t,default:l}}=e;return{value:n,label:a,attributes:t,default:l}}))}(a);return function(e){const n=(0,r.l)(e,((e,n)=>e.value===n.value));if(n.length>0)throw new Error(`Docusaurus error: Duplicate values "${n.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[n,a])}function h(e){let{value:n,tabValues:a}=e;return a.some((e=>e.value===n))}function u(e){let{queryString:n=!1,groupId:a}=e;const i=(0,l.k6)(),r=function(e){let{queryString:n=!1,groupId:a}=e;if("string"==typeof n)return n;if(!1===n)return null;if(!0===n&&!a)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return a??null}({queryString:n,groupId:a});return[(0,s._X)(r),(0,t.useCallback)((e=>{if(!r)return;const n=new URLSearchParams(i.location.search);n.set(r,e),i.replace({...i.location,search:n.toString()})}),[r,i])]}function p(e){const{defaultValue:n,queryString:a=!1,groupId:l}=e,s=o(e),[r,d]=(0,t.useState)((()=>function(e){let{defaultValue:n,tabValues:a}=e;if(0===a.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(n){if(!h({value:n,tabValues:a}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${n}" but none of its children has the corresponding value. Available values are: ${a.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return n}const t=a.find((e=>e.default))??a[0];if(!t)throw new Error("Unexpected error: 0 tabValues");return t.value}({defaultValue:n,tabValues:s}))),[p,m]=u({queryString:a,groupId:l}),[x,j]=function(e){let{groupId:n}=e;const a=function(e){return e?`docusaurus.tab.${e}`:null}(n),[l,i]=(0,c.Nk)(a);return[l,(0,t.useCallback)((e=>{a&&i.set(e)}),[a,i])]}({groupId:l}),f=(()=>{const e=p??x;return h({value:e,tabValues:s})?e:null})();(0,i.Z)((()=>{f&&d(f)}),[f]);return{selectedValue:r,selectValue:(0,t.useCallback)((e=>{if(!h({value:e,tabValues:s}))throw new Error(`Can't select invalid tab value=${e}`);d(e),m(e),j(e)}),[m,j,s]),tabValues:s}}},42658:(e,n,a)=>{a.d(n,{Z:()=>t});const t=a.p+"assets/images/call_edit_screenshot-c3bf18fe90b9da9dd15ee97b1e4c0d19.png"},64850:(e,n,a)=>{a.d(n,{Z:()=>t});const t=a.p+"assets/images/export_modal-e287b61a389e218bb79d598089bde752.png"},61276:(e,n,a)=>{a.d(n,{Z:()=>t});const t=a.p+"assets/images/basic_call-d333612aef700095ccc45371891486e8.png"},11151:(e,n,a)=>{a.d(n,{Z:()=>r,a:()=>s});var t=a(67294);const l={},i=t.createContext(l);function s(e){const n=t.useContext(i);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(l):e.components||l:s(e.components),t.createElement(i.Provider,{value:n},e.children)}}}]);