var O=Object.defineProperty;var Q=(s,r,i)=>r in s?O(s,r,{enumerable:!0,configurable:!0,writable:!0,value:i}):s[r]=i;var S=(s,r,i)=>(Q(s,typeof r!="symbol"?r+"":r,i),i);import{d as H,o as d,c as h,e as B,a as c,t as y,l as P,r as x,A as Y,s as Z,B as M,f as ee,u as $,n as te,F as V,b as j,w as se,v as re,C as ne,D as ae,E as q,G as K,g as oe,h as z,y as I,z as ie}from"./index-a8e68aff.js";import{u as G,w as le,a as ue,_ as ce,c as me}from"./index-5dd877b8.js";const de={key:0,class:"text-rose-400 text-lg",name:"alert-circle"},he={key:1,label:"Loading",indeterminate:"",static:"white",size:"s"},fe=["progress"],J=H({__name:"TheChatMessageStatus",props:{status:null},setup(s){return(r,i)=>s.status.kind==="failure"?(d(),h("ion-icon",de)):s.status.kind==="loading"?(d(),h("sp-progress-circle",he)):s.status.kind==="progress"?(d(),h("sp-progress-circle",{key:2,label:"Loading",static:"white",progress:s.status.progress,size:"s"},null,8,fe)):B("",!0)}}),ge={class:"message-element message-element-model my-2"},pe={class:"flex items-center gap-1 message__model-title"},_e=c("ion-icon",{name:"school"},null,-1),ye={class:"text-xs"},ve={class:"text-left message__content"},xe=H({__name:"TheChatModelMessage",props:{message:null},setup(s){return(r,i)=>(d(),h("li",ge,[c("div",pe,[_e,c("span",ye,y(s.message.title),1),P(J,{status:s.message.status},null,8,["status"])]),c("output",null,[c("pre",ve,y(s.message.message.trim()),1)])]))}});function X(s){const r=[];if(!s.startsWith("/"))return;const i=s.split(" ");r.push({type:"cmd",value:i[0].substring(1)});for(let m=1;m<i.length;m++){const u=i[m];if(u.startsWith("'")||u.startsWith('"')){const f=u.slice(1,-1);r.push({type:"string",value:f})}else if(u.includes("=")){const[f,p]=u.split("=");p.startsWith("'")||p.startsWith('"')?r.push({type:"arg",name:f,value:p.slice(1,-1)}):r.push({type:"arg",name:f,value:p})}else r.push({type:"string",value:u})}return r}class be{constructor(r=100){S(this,"_history");S(this,"_maxHistoryLength",100);S(this,"_top",0);S(this,"_bottom",0);S(this,"_iterCurr",0);this._maxHistoryLength=r,this._history=new Array(this._maxHistoryLength)}get history(){return this._history}get maxHistoryLength(){return this._maxHistoryLength}get size(){return this._top>this._bottom?this._top-this._bottom:this._maxHistoryLength-this._bottom+this._top}addHistory(r){this._history[this._top]=r,this._top=(this._top+1)%this._maxHistoryLength,this._top==this._bottom&&(this._bottom=(this._bottom+1)%this._maxHistoryLength),this._history[this._top]=r,this._iterCurr=this._top}getPreviousHistory(){return this._iterCurr==this._bottom?this._history[this._iterCurr]:(this._iterCurr=(this._maxHistoryLength+this._iterCurr-1)%this._maxHistoryLength,this._iterCurr==this._bottom?this._history[this._iterCurr]:this._history[this._iterCurr])}getNextHistory(){if(this._iterCurr!=this._top&&(this._iterCurr=(this._iterCurr+1)%this._maxHistoryLength,this._iterCurr!=this._top))return this._history[this._iterCurr]}}const we={class:"h-full w-full",style:{display:"grid","grid-template-rows":"1fr auto"}},ke=["onSubmit"],Ce=c("sp-field-label",{for:"message-box"},"Message",-1),Se=["value"],$e={class:"flex flex-grow items-center gap-2"},Me=c("button",{type:"submit",class:"flex aspect-square w-10 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-zinc-100 transition-colors hover:bg-zinc-700 active:bg-zinc-900","aria-label":"Send Message",title:"Send Message"},[c("ion-icon",{name:"send","aria-hidden":"true"})],-1),He=H({__name:"TheChatProvider",emits:["message","command"],setup(s,{expose:r,emit:i}){const m=x(null),u=x(""),f=new be;function p(){if(!m.value)return;const t=m.value.value.trim();if(!t)return;const a=X(t);a?i("command",a):i("message",t),f.addHistory(t),u.value=""}function A(t){if(t.key==="Enter"&&!t.shiftKey)t.preventDefault(),p();else if(t.key==="ArrowUp"&&t.shiftKey){t.preventDefault();const a=f.getPreviousHistory();u.value=a??""}else if(t.key==="ArrowDown"&&t.shiftKey){t.preventDefault();const a=f.getNextHistory();u.value=a??""}}function C(){var t;return m.value?(t=m.value.shadowRoot)==null?void 0:t.querySelector("textarea"):null}Y(async t=>{if(!m.value)return;const a=setInterval(async()=>{if(!m.value)return;await ae();const o=C();o&&(o.style.maxHeight="15rem",o.style.overflow="auto",clearInterval(a))},200);t(()=>{clearInterval(a)})});const v=x(0),g=x([]),{commands:L}=Z(G()),b=x(!1),R=M(()=>{if(!m.value)return 0;const t=getComputedStyle(m.value).fontSize;return parseFloat(t)*.5});le(u,()=>{const t=C();if(!t)return;v.value=t.selectionStart??0;const a=u.value;if(!a.startsWith("/")){b.value=!1,g.value=[];return}const o=X(a);if(!o){g.value=[],b.value=!1;return}b.value=!0,o[0].value!==""&&(g.value=o)});const T=M(()=>{if(!b.value)return[];if(g.value.length===0)return L.value.map(o=>({type:"cmd",value:o.name}));const t=L.value.find(o=>o.name===g.value[0].value);if(!t)return[];const a=[];for(let o=0;o<t.args.length;++o){const w=t.args[o],_=w.name;g.value.some(D=>D.type==="arg"?D.name===_:D.value===_)||(w.type==="boolean"?a.push({type:"string",value:_}):a.push({type:"arg",name:_,value:""}))}return a});function F(t){return t.type==="cmd"?`/${t.value}`:t.type==="arg"?`${t.name}=`:`${t.value}`}const e=x(null),n={isAtBottom:!0,threshold:.02};function l(){requestAnimationFrame(()=>{if(!e.value)return;const t=e.value,a=t.scrollHeight,o=t.scrollTop,w=t.clientHeight;a-o-w<=a*n.threshold?n.isAtBottom=!0:n.isAtBottom=!1})}function W(t){requestAnimationFrame(()=>{const a=e.value;!a||!n.isAtBottom||a.lastElementChild.scrollIntoView(t)})}return r({scrollToLatestMessage:W}),(t,a)=>(d(),h("div",we,[c("article",{class:"w-full overflow-y-auto px-4 pt-2 flex flex-col last:mb-8",ref_key:"messageContainerRef",ref:e,onScroll:l},[ee(t.$slots,"default")],544),c("form",{class:"w-full p-2 relative",onSubmit:ne(p,["prevent"])},[Ce,$(T).length>0&&b.value?(d(),h("sp-popover",{key:0,open:"",style:te([{position:"absolute",bottom:"100%"},{left:`min(${v.value*$(R)}px, 100%)`}]),class:"border border-zinc-600 max-h-[10rem]"},[c("sp-menu",null,[(d(!0),h(V,null,j($(T),o=>(d(),h("sp-menu-item",{key:o.value+o.type,value:o.value},y(F(o)),9,Se))),128))])],4)):B("",!0),c("div",$e,[se(c("sp-textfield",{grows:"",multiline:"",class:"w-full max-h-[15rem] min-w-[70%] overflow-hidden resize-none",placeholder:"Write a message...",onKeydown:A,ref_key:"inputRef",ref:m,"onUpdate:modelValue":a[0]||(a[0]=o=>u.value=o)},null,544),[[re,u.value,void 0,{trim:!0}]]),Me])],40,ke)]))}}),ze={class:"flex items-center gap-2 font-mono"},Ae=["data-kind"],Le={class:"flex-grow"},Te={class:"text-xs font-semibold text-slate-400"},Ne={key:0},Ee={class:"message__content font-mono"},Be=["progress"],Re=H({__name:"TheChatSystemMessage",props:{message:null},setup(s){const r=s,i=M(()=>{const{message:u}=r;return u.kind==="progress"?void 0:u.message.trim()}),m=M(()=>{const{message:u}=r;return u.kind==="progress"?Math.round(u.progress):void 0});return(u,f)=>(d(),h("li",ze,[c("span",{class:"system-message-tag","data-kind":s.message.kind},[q(" ["),c("span",Le,y(s.message.kind),1),q("] ")],8,Ae),c("span",Te,"<"+y(s.message.function_name)+">",1),s.message.kind!=="progress"?(d(),h("output",Ne,[c("pre",Ee,y($(i)),1)])):(d(),h("sp-meter",{key:1,progress:$(m),class:"my-2 w-full",static:"white"},y(s.message.message),9,Be))]))}}),Fe={class:"message-element message-element-right message-element-user my-2"},We={class:"flex items-center gap-1 message__user-title"},De=c("ion-icon",{name:"person"},null,-1),Ie={class:"text-xs"},Ve={class:"text-left message__content"},Pe=H({__name:"TheChatUserMessage",props:{message:null},setup(s){return(r,i)=>(d(),h("li",Fe,[c("div",We,[P(J,{status:s.message.status},null,8,["status"]),De,c("span",Ie,y(s.message.title),1)]),c("output",null,[c("pre",Ve,y(s.message.message.trim()),1)])]))}});var k=256,E=[],N;for(;k--;)E[k]=(k+256).toString(16).substring(1);function Ue(){var s=0,r,i="";if(!N||k+16>256){for(N=Array(s=256);s--;)N[s]=256*Math.random()|0;s=k=0}for(;s<16;s++)r=N[k+s],s==6?i+=E[r&15|64]:s==8?i+=E[r&63|128]:i+=E[r],s&1&&s>1&&s<11&&(i+="-");return k++,i}const je=H({__name:"ChatView",setup(s){const r=x([]),i=ie(),m=x(null),u=new Set,f=M(()=>{const{query:e}=i.currentRoute.value,n=Array.isArray(e.path)?e.path[0]:e.path;return n?String(n):null});K(()=>{f.value||i.replace({name:"error",params:{message:"Model path is not specified"}})});function p(e){var n;u.has(e.id)||(u.add(e.id),r.value.push(e),(n=m.value)==null||n.scrollToLatestMessage())}function A(e){for(let n=r.value.length-1;n>=0;--n)if(r.value[n].id===e)return n;return-1}function C(e){var l;if(!u.has(e.id)){p(e);return}const n=A(e.id);r.value[n]=e,(l=m.value)==null||l.scrollToLatestMessage()}const v=ue(),g=M(()=>{const{query:e}=i.currentRoute.value,n=Array.isArray(e.model_params)?e.model_params[0]:e.model_params;return n?JSON.parse(String(n)):null});function L(e){switch(e.type){case"model-message":case"system-message":case"user-message":{C(e);break}case"message-ack":{const n=A(e.webui_id);if(n===-1)return;const l=r.value[n];if(l.type!=="user-message")return;l.status={kind:"loading"},l.id=e.id,u.add(e.id);break}case"reset-messages":{const n=e.messages;u.clear(),r.value=[],n.forEach(C);break}default:return}}K(async()=>{if(!v.isConnected)await i.replace({name:"error",params:{message:"Websocket is not connected"}});else{if(g.value)return;await i.replace({name:"error",params:{message:"Model params is not specified"}})}}),oe(()=>{g.value&&(v.on(L),v.message({type:"init-model",model_path:f.value,...g.value}))});function b(e){const n=Ue(),l={type:"user-message",message:e,webui_id:n,id:n,title:"user",status:{kind:"loading"}};C(l),v.message(l)}function R(e,n){switch(n){case"float":{if(e==null)return 0;const l=parseFloat(e);return Number.isNaN(l)?0:l}case"int":{if(e==null)return 0;const l=parseInt(e);return Number.isNaN(l)?0:l}case"string":return String(e??"");case"boolean":return e==null||e==="true";default:return e}}function T(e){if(e[0].type!=="cmd")return;const n=e[0].value,l=[],{commands:W}=G(),t=W.find(a=>a.name===n);if(t){for(let a=1;a<e.length;++a){const o=e[a];if(o.type==="cmd")return;const w=o.type==="arg"?o.name:o.value,_=t.args.find(U=>U.name===w);_&&l.push({name:w,value:R(o.value,_.type),type:_.type})}return{name:n,args:l}}}function F(e){const n=T(e);n&&v.message({type:"invoke-command",command:n.name,args:n.args})}return(e,n)=>(d(),z(ce,null,{aside:I(()=>[P(me)]),default:I(()=>[$(f)?(d(),z(He,{key:0,ref_key:"chatProviderRef",ref:m,onMessage:b,onCommand:F},{default:I(()=>[(d(!0),h(V,null,j(r.value,l=>(d(),h(V,{key:l.id},[l.type==="system-message"?(d(),z(Re,{key:0,message:l},null,8,["message"])):l.type==="user-message"?(d(),z(Pe,{key:1,message:l},null,8,["message"])):l.type==="model-message"?(d(),z(xe,{key:2,message:l},null,8,["message"])):B("",!0)],64))),128))]),_:1},512)):B("",!0)]),_:1}))}});export{je as default};
