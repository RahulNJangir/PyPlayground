import threading
import traceback
import webbrowser
import subprocess
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
import torch

try:
    import psutil
except ImportError:
    psutil = None

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HOST, PORT = "127.0.0.1", 5000

print("\n" + "=" * 55)
print("                  NICK AI CHATBOT")
print("=" * 55)
print("\nChecking GPU...")

if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    print("CUDA: Available")
    print("GPU:", GPU_NAME)
else:
    DEVICE = "cpu"
    GPU_NAME = "CPU"
    print("CUDA: Not available")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nLoading Llama 3.2 1B...")
if DEVICE == "cuda":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.to("cpu")
model.eval()

print("\n" + "=" * 55)
print("MODEL READY")
print("=" * 55)

app = Flask(__name__)

MODEL_STATUS = "Ready"
MODEL_BUSY = False
CHATS = {}
CHAT_COUNTER = 0

TOKEN_LOGS = []

def record_token_usage(count):
    TOKEN_LOGS.append({'timestamp': datetime.now(), 'tokens': count})

def get_hourly_token_count():
    now = datetime.now()
    cutoff = now - timedelta(hours=1)
    global TOKEN_LOGS
    TOKEN_LOGS = [entry for entry in TOKEN_LOGS if entry['timestamp'] > cutoff]
    return sum(entry['tokens'] for entry in TOKEN_LOGS)

def get_hourly_history():
    now = datetime.now()
    hourly_buckets = [0] * 12
    for entry in TOKEN_LOGS:
        delta = now - entry['timestamp']
        hours_ago = int(delta.total_seconds() // 3600)
        if 0 <= hours_ago < 12:
            hourly_buckets[11 - hours_ago] += entry['tokens']
    return hourly_buckets

def new_chat_record():
    global CHAT_COUNTER
    CHAT_COUNTER += 1
    chat_id = f"chat_{CHAT_COUNTER}_{int(time.time())}"
    record = {
        "id": chat_id,
        "name": f"New Chat {CHAT_COUNTER}",
        "messages": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "updated": datetime.now().isoformat(timespec='seconds')
    }
    CHATS[chat_id] = record
    return record

def get_or_create_chat(chat_id):
    if chat_id and chat_id in CHATS:
        return CHATS[chat_id]
    return new_chat_record()

default_chat = new_chat_record()

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NICK AI</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;width:100%;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);transition:background 0.25s ease, color 0.25s ease}

:root{
 --bg:#f8fafc;--side:#0f172a;--side2:#1e293b;--border:#e2e8f0;--text:#0f172a;--muted:#64748b;
 --input:#ffffff;--user:#f1f5f9;--card:#ffffff;--accent:#3b82f6;--accent-glow:rgba(59,130,246,0.15);
 --shadow:0 10px 30px -5px rgba(0,0,0,0.05);
}

body.dark{
 --bg:#090d16;--side:#050811;--side2:#0d1322;--border:#1e293b;--text:#f8fafc;--muted:#94a3b8;
 --input:#0f172a;--user:#111827;--card:#0d1322;--accent:#3b82f6;--accent-glow:rgba(59,130,246,0.25);
 --shadow:0 10px 30px -5px rgba(0,0,0,0.5);
}

.app{height:100vh;display:flex;overflow:hidden}

.sidebar{
  width:310px;
  background:linear-gradient(180deg,var(--side),var(--side2));
  border-right:1px solid rgba(255,255,255,.08);
  padding:18px;
  display:flex;
  flex-direction:column;
  flex-shrink:0;
  color:#fff;
  height:100vh;
  overflow-y:auto;
  scrollbar-width:thin;
  scrollbar-color:rgba(255,255,255,0.15) transparent;
}
.sidebar::-webkit-scrollbar{width:5px}
.sidebar::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.15);border-radius:4px}

.logo{display:flex;align-items:center;gap:12px;padding:4px 4px 18px;font-size:20px;font-weight:800;letter-spacing:-.01em}
.logoIcon{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:#fff;display:grid;place-items:center;font-weight:900;box-shadow:0 4px 15px rgba(59,130,246,0.4)}

button{font:inherit;outline:none}
.newChat{width:100%;height:44px;border:1px solid rgba(255,255,255,.12);border-radius:10px;background:rgba(255,255,255,.06);color:#fff;cursor:pointer;text-align:left;padding:0 14px;font-weight:600;display:flex;align-items:center;gap:8px;transition:all 0.2s}
.newChat:hover{background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.2)}

.label{font-size:10px;color:#94a3b8;font-weight:800;margin:18px 4px 8px;letter-spacing:.12em;text-transform:uppercase}
.card{border:1px solid rgba(255,255,255,.08);border-radius:12px;background:rgba(255,255,255,.03);padding:12px 14px;box-shadow:0 4px 20px rgba(0,0,0,.15)}

.model{font-size:13px;font-weight:700;margin-bottom:6px;color:#fff}.status{font-size:11px;color:#94a3b8;display:flex;gap:7px;align-items:center}.dot{width:7px;height:7px;border-radius:50%;background:#10b981;box-shadow:0 0 8px #10b981}

.row{display:flex;justify-content:space-between;gap:8px;padding:3px 0;font-size:12px}.row span:first-child{color:#94a3b8}.row span:last-child{font-weight:700;text-align:right;color:#f8fafc}
.meter{height:5px;background:rgba(255,255,255,.08);border-radius:10px;overflow:hidden;margin:4px 0 6px}.meter i{display:block;height:100%;width:0;border-radius:10px;background:linear-gradient(90deg,#3b82f6,#8b5cf6);transition:width .35s ease}

.graphCanvas{width:100%;height:90px;margin-top:6px;display:block}

/* FIXED CHAT HISTORY SIDEBAR VISIBILITY */
.history{
  min-height:100px;
  max-height:160px;
  overflow-y:auto;
  display:flex;
  flex-direction:column;
  gap:6px;
  padding:6px;
}
.historyItem{display:flex;align-items:center;justify-content:space-between;width:100%;border:0;background:rgba(255,255,255,0.02);color:#cbd5e1;text-align:left;padding:8px 10px;border-radius:8px;cursor:pointer;font-size:12px;transition:all 0.15s;border:1px solid rgba(255,255,255,0.05)}
.historyItem:hover{background:rgba(255,255,255,0.08);color:#fff}
.historyItem.active{background:rgba(59,130,246,.25);color:#fff;font-weight:600;border-color:rgba(59,130,246,0.4)}
.historyText{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.historyDelete{color:#64748b;border:0;background:transparent;cursor:pointer;padding:2px 6px;font-size:14px;transition:color 0.15s}
.historyDelete:hover{color:#ef4444}

.controls{display:grid;gap:12px}.controlHead{display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#cbd5e1;margin-bottom:4px}.controlHead b{color:#fff}
input[type=range]{width:100%;accent-color:#3b82f6;cursor:pointer}

.sideBtn{width:100%;padding:10px 12px;border:1px solid rgba(255,255,255,.1);border-radius:10px;background:transparent;color:#e2e8f0;cursor:pointer;text-align:left;margin-top:8px;font-size:12px;transition:all 0.2s}
.sideBtn:hover{background:rgba(255,255,255,.08)}
.bottom{margin-top:auto;padding-top:10px}

.main{flex:1;min-width:0;display:flex;flex-direction:column;background:var(--bg)}
.top{height:68px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 28px;background:var(--bg);backdrop-filter:blur(12px)}
.title{font-size:17px;font-weight:800;letter-spacing:-.01em}.title span{color:var(--accent)}

.tokens{display:flex;gap:8px;align-items:center}
.tokenPill{padding:6px 12px;border:1px solid var(--border);border-radius:8px;background:var(--card);font-size:11px;color:var(--muted);box-shadow:var(--shadow);display:flex;align-items:center;gap:4px}
.tokenPill b{color:var(--text)}
.tokenPill.highlight{border-color:var(--accent);background:var(--accent-glow)}
.tokenPill.live{border-color:#10b981;background:rgba(16,185,129,0.12);color:#10b981}

.chat{flex:1;overflow-y:auto;padding-bottom:20px}
.welcome{max-width:800px;margin:auto;padding:80px 24px 30px;text-align:center}
.welcomeIcon{width:72px;height:72px;margin:0 auto 20px;border-radius:20px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:#fff;display:grid;place-items:center;font-size:32px;font-weight:900;box-shadow:0 12px 30px rgba(59,130,246,0.3)}
.welcome h1{font-size:32px;margin:0 0 10px;letter-spacing:-.03em;font-weight:800}.welcome p{color:var(--muted);line-height:1.6;margin:0 auto 32px;max-width:600px;font-size:15px}

.suggestions{display:grid;grid-template-columns:1fr 1fr;gap:12px;text-align:left}
.suggestion{padding:16px;border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:12px;cursor:pointer;box-shadow:var(--shadow);transition:all 0.2s}
.suggestion:hover{transform:translateY(-2px);border-color:var(--accent)}

.message{border-bottom:1px solid var(--border)}.message.user{background:var(--user)}
.inner{max-width:900px;margin:auto;padding:22px 24px;display:flex;gap:16px}
.avatar{width:32px;height:32px;border-radius:10px;display:grid;place-items:center;flex:none;font-size:12px;font-weight:800;color:#fff}
.user .avatar{background:#64748b}.assistant .avatar{background:linear-gradient(135deg,#3b82f6,#8b5cf6)}
.content{flex:1;min-width:0;line-height:1.7;font-size:15px}

.metaHeader{display:flex;align-items:center;gap:10px;margin-bottom:6px;font-size:11px;color:var(--muted)}
.msgBadge{padding:2px 7px;border-radius:6px;background:rgba(59,130,246,0.12);color:var(--accent);font-weight:700;font-size:10px;border:1px solid rgba(59,130,246,0.2)}
.msgText{white-space:pre-wrap;word-break:break-word}

.copy{border:0;background:none;color:var(--muted);cursor:pointer;font-size:11px;padding:6px 0;display:inline-block;margin-top:6px}
.copy:hover{color:var(--accent)}

.thinking{display:flex;gap:6px;padding:8px 0}.thinking i{width:8px;height:8px;background:var(--accent);border-radius:50%;animation:b 1.2s infinite}.thinking i:nth-child(2){animation-delay:.15s}.thinking i:nth-child(3){animation-delay:.3s}@keyframes b{0%,60%,100%{transform:translateY(0);opacity:.35}30%{transform:translateY(-6px);opacity:1}}

.inputArea{padding:16px 24px 24px}.inputWrap{max-width:900px;margin:auto;position:relative}
.input{width:100%;min-height:60px;max-height:180px;resize:none;border:1px solid var(--border);border-radius:16px;background:var(--input);color:var(--text);padding:18px 60px 18px 18px;outline:none;font:15px inherit;box-shadow:var(--shadow);transition:all 0.2s}
.input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow),var(--shadow)}
.send{position:absolute;right:10px;bottom:10px;width:40px;height:40px;border:0;border-radius:10px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:#fff;cursor:pointer;font-size:18px;display:grid;place-items:center;box-shadow:0 4px 15px rgba(59,130,246,0.3);transition:all 0.2s}
.send:disabled{opacity:.4;cursor:not-allowed}
.note{max-width:900px;margin:8px auto 0;text-align:center;color:var(--muted);font-size:11px}

.liveStatus{display:flex;align-items:center;gap:6px;font-size:10px;color:#94a3b8;margin-top:8px}
.liveDot{width:6px;height:6px;border-radius:50%;background:#10b981;box-shadow:0 0 8px #10b981}

@media(max-width:900px){.sidebar{width:260px}}
@media(max-width:760px){.sidebar{display:none}.top{padding:0 16px}.tokens{gap:4px}.welcome{padding-top:40px}.suggestions{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="app">
<aside class="sidebar">
<div class="logo"><div class="logoIcon">N</div><span>NICK AI</span></div>
<button class="newChat" onclick="newChat()"><span>＋</span> New chat</button>

<div class="label">CHAT HISTORY</div>
<div class="card history" id="history"></div>

<div class="label">HOURLY TOKEN USAGE</div>
<div class="card">
  <div class="row"><span>Current Hour</span><b id="sideHourlyVal">0 tok</b></div>
  <canvas id="tokenChart" class="graphCanvas"></canvas>
</div>

<div class="label">MODEL</div>
<div class="card">
  <div class="model">Llama 3.2 1B Instruct</div>
  <div class="status"><span class="dot"></span><span id="modelStatus">Local • Ready</span></div>
</div>

<div class="label">LIVE SYSTEM MONITOR</div>
<div class="card">
  <div class="row"><span>CPU</span><span id="cpu">0%</span></div>
  <div class="meter"><i id="cpuBar"></i></div>
  <div class="row"><span>RAM</span><span id="ram">0 / 0 GB</span></div>
  <div class="meter"><i id="ramBar"></i></div>
  <div class="row"><span>GPU</span><span id="gpu">...</span></div>
  <div class="row"><span>GPU Usage</span><span id="gpuUse">0%</span></div>
  <div class="meter"><i id="gpuBar"></i></div>
  <div class="row"><span>VRAM</span><span id="vram">0 / 0 GB</span></div>
  <div class="meter"><i id="vramBar"></i></div>
  <div class="row"><span>DISK</span><span id="disk">0 / 0 GB</span></div>
  <div class="meter"><i id="diskBar"></i></div>
  <div class="row"><span>Disk Free</span><span id="diskFree">0 GB</span></div>
  <div class="liveStatus"><span class="liveDot"></span>Live telemetry active</div>
</div>

<div class="label">AI CONTROLS</div>
<div class="card controls">
  <div>
    <div class="controlHead"><span>Temperature</span><b id="tempValue">0.70</b></div>
    <input id="temperature" type="range" min="0.1" max="1.5" step="0.05" value="0.70" oninput="showControl('temperature','tempValue',2)">
  </div>
  <div>
    <div class="controlHead"><span>Top-P</span><b id="topPValue">0.90</b></div>
    <input id="topP" type="range" min="0.1" max="1" step="0.05" value="0.90" oninput="showControl('topP','topPValue',2)">
  </div>
  <div>
    <div class="controlHead"><span>Max tokens</span><b id="maxTokensValue">200</b></div>
    <input id="maxTokens" type="range" min="32" max="512" step="16" value="200" oninput="showControl('maxTokens','maxTokensValue',0)">
  </div>
  <div>
    <div class="controlHead"><span>Repetition penalty</span><b id="repValue">1.05</b></div>
    <input id="repetitionPenalty" type="range" min="1" max="1.5" step="0.01" value="1.05" oninput="showControl('repetitionPenalty','repValue',2)">
  </div>
</div>

<div class="bottom">
  <button class="sideBtn" onclick="resetControls()">↺ Reset parameters</button>
  <button class="sideBtn" onclick="theme()">☾ Toggle theme</button>
</div>
</aside>

<main class="main">
<header class="top">
  <div class="title">NICK <span>AI</span></div>
  <div class="tokens">
    <span class="tokenPill live">TYPING: <b id="typingTokens">0 tok</b></span>
    <span class="tokenPill">IN <b id="ti">0</b></span>
    <span class="tokenPill">OUT <b id="to">0</b></span>
    <span class="tokenPill">TOTAL <b id="tt">0</b></span>
    <span class="tokenPill highlight">1 HR USAGE: <b id="hourlyTokens">0</b></span>
    <span class="tokenPill">SPEED <b id="topSpeed">0 tok/s</b></span>
  </div>
</header>

<section class="chat" id="chat">
  <div class="welcome" id="welcome">
    <div class="welcomeIcon">N</div>
    <h1>How can I help you?</h1>
    <p>NICK AI runs locally with Llama 3.2 1B Instruct. Private, instant, and unlimited generation.</p>
    <div class="suggestions">
      <button class="suggestion" onclick="suggest('Explain artificial intelligence in simple words.')">Explain AI simply</button>
      <button class="suggestion" onclick="suggest('Give me 5 Linux interview questions for a fresher.')">Linux interview questions</button>
      <button class="suggestion" onclick="suggest('Explain AWS EC2 in simple words.')">Explain AWS EC2</button>
      <button class="suggestion" onclick="suggest('Write a small Python program and explain it.')">Help me with Python</button>
    </div>
  </div>
</section>

<div class="inputArea">
  <div class="inputWrap">
    <textarea id="input" class="input" rows="1" placeholder="Message NICK AI..."></textarea>
    <button id="send" class="send" onclick="send()">↑</button>
  </div>
  <div class="note">NICK AI • Local Llama 3.2 1B • On-Device Inference</div>
</div>
</main>
</div>

<script>
let currentChatId=null,messages=[],tin=0,tout=0,ttotal=0,hourlyTotal=0,lastSpeed=0,hourlyHistory=[0,0,0,0,0,0,0,0,0,0,0,0];
const input=document.getElementById('input'),sendBtn=document.getElementById('send'),chat=document.getElementById('chat');

function estimateTokens(text) {
  if (!text || !text.trim()) return 0;
  return Math.max(1, Math.ceil(text.trim().length / 3.8));
}

function update(){
  document.getElementById('ti').textContent=tin.toLocaleString();
  document.getElementById('to').textContent=tout.toLocaleString();
  document.getElementById('tt').textContent=ttotal.toLocaleString();
  document.getElementById('hourlyTokens').textContent=hourlyTotal.toLocaleString();
  document.getElementById('sideHourlyVal').textContent=hourlyTotal.toLocaleString()+' tok';
  document.getElementById('topSpeed').textContent=lastSpeed.toFixed(1)+' tok/s';
  drawChart();
}

function drawChart(){
  const canvas=document.getElementById('tokenChart'), ctx=canvas.getContext('2d');
  canvas.width=canvas.offsetWidth*2; canvas.height=canvas.offsetHeight*2;
  ctx.scale(2,2);
  const w=canvas.offsetWidth, h=canvas.offsetHeight;
  ctx.clearRect(0,0,w,h);
  
  const max=Math.max(...hourlyHistory, 100);
  const step=w/(hourlyHistory.length-1);

  ctx.beginPath();
  ctx.moveTo(0, h);
  hourlyHistory.forEach((val, i)=>{
    const x=i*step;
    const y=h - (val/max)*(h-15) - 5;
    if(i===0) ctx.lineTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(w, h);
  ctx.fillStyle='rgba(59, 130, 246, 0.15)';
  ctx.fill();

  ctx.beginPath();
  hourlyHistory.forEach((val, i)=>{
    const x=i*step;
    const y=h - (val/max)*(h-15) - 5;
    if(i===0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle='#3b82f6';
  ctx.lineWidth=2;
  ctx.stroke();
}

function showControl(id,label,decimals){
  const v=Number(document.getElementById(id).value);
  document.getElementById(label).textContent=decimals? v.toFixed(decimals):Math.round(v);
}

function resetControls(){
  document.getElementById('temperature').value=.70;
  document.getElementById('topP').value=.90;
  document.getElementById('maxTokens').value=200;
  document.getElementById('repetitionPenalty').value=1.05;
  showControl('temperature','tempValue',2);
  showControl('topP','topPValue',2);
  showControl('maxTokens','maxTokensValue',0);
  showControl('repetitionPenalty','repValue',2);
}

function controlValues(){
  return {
    temperature:Number(document.getElementById('temperature').value),
    top_p:Number(document.getElementById('topP').value),
    max_tokens:Number(document.getElementById('maxTokens').value),
    repetition_penalty:Number(document.getElementById('repetitionPenalty').value)
  };
}

function usageClass(v){
  if(v < 60) return 'good';
  if(v < 80) return 'warn';
  return 'bad';
}
function applyStatus(el,v){
  const c=usageClass(v);
  el.style.color=c==='good'?'#10b981':c==='warn'?'#f59e0b':'#ef4444';
}
function applyBar(el,v){
  const c=usageClass(v);
  el.style.width=Math.min(Math.max(v,0),100)+'%';
  el.style.background=c==='good'?'#10b981':c==='warn'?'#f59e0b':'#ef4444';
}

async function system(){
  try{
    let r=await fetch('/system',{cache:'no-store'}),d=await r.json();
    document.getElementById('cpu').textContent=d.cpu+'%';
    applyStatus(document.getElementById('cpu'),d.cpu); applyBar(document.getElementById('cpuBar'),d.cpu);

    document.getElementById('ram').textContent=d.ram_used+' / '+d.ram_total+' GB';
    applyStatus(document.getElementById('ram'),d.ram_percent); applyBar(document.getElementById('ramBar'),d.ram_percent);

    document.getElementById('gpu').textContent=d.gpu;
    document.getElementById('gpuUse').textContent=d.gpu_usage+'%';
    applyStatus(document.getElementById('gpuUse'),d.gpu_usage); applyBar(document.getElementById('gpuBar'),d.gpu_usage);

    document.getElementById('vram').textContent=d.vram_used+' / '+d.vram_total+' GB';
    applyStatus(document.getElementById('vram'),d.vram_percent); applyBar(document.getElementById('vramBar'),d.vram_percent);

    document.getElementById('disk').textContent=d.disk_used+' / '+d.disk_total+' GB';
    document.getElementById('diskFree').textContent=d.disk_free+' GB';
    applyStatus(document.getElementById('disk'),d.disk_percent); applyBar(document.getElementById('diskBar'),d.disk_percent);

    document.getElementById('modelStatus').textContent=d.model_status+' • '+d.device;
    
    hourlyTotal = d.hourly_tokens || 0;
    if(d.hourly_history) hourlyHistory = d.hourly_history;
    update();
  }catch(e){}
}
setInterval(system,1000);

function add(role,text,timeStr,tokenCount){
  let w=document.getElementById('welcome');if(w)w.remove();
  let m=document.createElement('div');m.className='message '+role;
  let inner=document.createElement('div');inner.className='inner';
  let av=document.createElement('div');av.className='avatar';av.textContent=role==='user'?'U':'N';
  let c=document.createElement('div');c.className='content';

  let timeVal = timeStr || new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  let meta = document.createElement('div');
  meta.className = 'metaHeader';
  meta.innerHTML = '<span>'+timeVal+'</span>' + (tokenCount ? '<span class="msgBadge">'+tokenCount+' tokens</span>' : '');
  
  let t=document.createElement('div');
  t.className = 'msgText';
  t.textContent=text;

  c.appendChild(meta);
  c.appendChild(t);

  if(role==='assistant'){
    let b=document.createElement('button');b.className='copy';b.textContent='Copy Response';
    b.onclick=()=>{navigator.clipboard.writeText(text);b.textContent='Copied!';setTimeout(()=>b.textContent='Copy Response',1200)};
    c.appendChild(b);
  }
  inner.append(av,c);m.appendChild(inner);chat.appendChild(m);bottom();
}

function think(){
  let m=document.createElement('div');m.id='thinking';m.className='message assistant';
  m.innerHTML='<div class="inner"><div class="avatar">N</div><div class="content"><div class="thinking"><i></i><i></i><i></i></div></div></div>';
  chat.appendChild(m);bottom();
}
function unthink(){let x=document.getElementById('thinking');if(x)x.remove()}
function bottom(){setTimeout(()=>chat.scrollTop=chat.scrollHeight,30)}

async function loadChats(){
  try{
    const r=await fetch('/chats',{cache:'no-store'}), d=await r.json();
    const box=document.getElementById('history');
    box.innerHTML='';
    if (d.chats && d.chats.length > 0) {
      if (!currentChatId) currentChatId = d.chats[0].id;
      d.chats.forEach(c=>{
        const row=document.createElement('div');
        row.className='historyItem '+(c.id===currentChatId?'active':'');
        row.innerHTML='<span class="historyText">'+c.name+'</span><button class="historyDelete" onclick="event.stopPropagation();deleteChat(\''+c.id+'\')">×</button>';
        row.onclick=()=>loadChat(c.id);
        box.appendChild(row);
      });
    } else {
      box.innerHTML = '<div style="font-size:11px;color:#64748b;padding:8px">No chat history</div>';
    }
  }catch(e){}
}

async function loadChat(id){
  const r=await fetch('/chats/'+encodeURIComponent(id));
  if(!r.ok)return;
  const d=await r.json();
  currentChatId=id;
  messages=d.chat.messages||[];
  tin=d.chat.input_tokens||0;
  tout=d.chat.output_tokens||0;
  ttotal=d.chat.total_tokens||0;
  lastSpeed=0;
  update();
  chat.innerHTML='';
  if(!messages.length){
    newChatScreen();
  }else{
    messages.forEach((m)=>add(m.role, m.content, m.timestamp, m.tokens));
  }
  loadChats();
}

function newChatScreen(){
  chat.innerHTML='<div class="welcome" id="welcome"><div class="welcomeIcon">N</div><h1>How can I help you?</h1><p>NICK AI runs locally with Llama 3.2 1B Instruct. Private, instant, and unlimited generation.</p><div class="suggestions">'+
  '<button class="suggestion" onclick="suggest(\'Explain artificial intelligence in simple words.\')">Explain AI simply</button>'+
  '<button class="suggestion" onclick="suggest(\'Give me 5 Linux interview questions for a fresher.\')">Linux interview questions</button>'+
  '<button class="suggestion" onclick="suggest(\'Explain AWS EC2 in simple words.\')">Explain AWS EC2</button>'+
  '<button class="suggestion" onclick="suggest(\'Write a small Python program and explain it.\')">Help me with Python</button></div></div>';
}

async function createChat(){
  const r=await fetch('/chats',{method:'POST'}),d=await r.json();
  currentChatId=d.chat.id; messages=[]; tin=tout=ttotal=0; lastSpeed=0;
  update(); newChatScreen(); loadChats();
}

async function deleteChat(id){
  await fetch('/chats/'+encodeURIComponent(id),{method:'DELETE'});
  if(id===currentChatId){currentChatId=null;messages=[];tin=tout=ttotal=0;lastSpeed=0;update();}
  await loadChats();
  if(!currentChatId) await createChat();
}

async function send(isRegenerate=false, regenerateText=''){
  let text=isRegenerate?regenerateText:input.value.trim();
  if(!text||sendBtn.disabled)return;
  
  let nowTime = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  let estTokens = estimateTokens(text);

  if(!isRegenerate){
    add('user', text, nowTime, estTokens);
    messages.push({role:'user', content:text, timestamp: nowTime, tokens: estTokens});
  }
  
  input.value='';
  input.style.height='60px';
  document.getElementById('typingTokens').textContent = '0 tok';
  sendBtn.disabled=true;
  think();

  try{
    let payload={chat_id:currentChatId,messages,...controlValues()};
    let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    let d=await r.json();unthink();
    if(!r.ok||d.error){add('assistant','Error: '+(d.error||'Unknown error'))}
    else{
      add('assistant', d.response, d.timestamp, d.output_tokens);
      messages.push({role:'assistant', content:d.response, timestamp: d.timestamp, tokens: d.output_tokens});
      
      // SYNC SESSION TOKENS FOR REAL-TIME DISPLAY
      tin = d.chat.input_tokens;
      tout = d.chat.output_tokens;
      ttotal = d.chat.total_tokens;
      lastSpeed = d.tokens_per_second || 0;
      hourlyTotal = d.hourly_tokens || hourlyTotal;
      
      update();
      loadChats();
    }
  }catch(e){unthink();add('assistant','Could not connect to server. Ensure Python backend is running.')}
  sendBtn.disabled=false;input.focus();
}

function suggest(x){input.value=x; document.getElementById('typingTokens').textContent = estimateTokens(x) + ' tok'; send()}
function newChat(){createChat();input.focus();}

function theme(){
  document.body.classList.toggle('dark');
  localStorage.setItem('nick-theme',document.body.classList.contains('dark')?'dark':'light');
}
if(localStorage.getItem('nick-theme')==='dark'||!localStorage.getItem('nick-theme')) document.body.classList.add('dark');

input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
input.addEventListener('input',()=>{
  input.style.height='auto';
  input.style.height=Math.min(input.scrollHeight,180)+'px';
  document.getElementById('typingTokens').textContent = estimateTokens(input.value) + ' tok';
});

system();update();loadChats();input.focus();
</script>
</body>
</html>'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/system')
def system_info():
    if psutil is not None:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used = mem.used / (1024 ** 3)
        ram_total = mem.total / (1024 ** 3)
        ram_percent = mem.percent
        try:
            disk = psutil.disk_usage("/")
        except Exception:
            disk = psutil.disk_usage("C:\\")
        disk_used = disk.used / (1024 ** 3)
        disk_free = disk.free / (1024 ** 3)
        disk_total = disk.total / (1024 ** 3)
        disk_percent = disk.percent
    else:
        cpu = ram_used = ram_total = ram_percent = 0
        disk_used = disk_free = disk_total = disk_percent = 0

    gpu_usage = 0.0
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_usage = float(result.stdout.strip().splitlines()[0])
        except Exception:
            gpu_usage = 0.0
    else:
        gpu = 'Not available'
        allocated = total_vram = 0

    vram_percent = (allocated / total_vram * 100) if total_vram else 0
    return jsonify(
        device=DEVICE.upper(),
        model_status='Busy' if MODEL_BUSY else MODEL_STATUS,
        cpu=round(cpu,1),
        ram_used=round(ram_used,2),
        ram_total=round(ram_total,2),
        ram_percent=round(ram_percent,1),
        gpu=gpu,
        gpu_usage=round(gpu_usage,1),
        vram_used=round(allocated,2),
        vram_total=round(total_vram,2),
        vram_percent=round(vram_percent,1),
        disk_used=round(disk_used,2),
        disk_free=round(disk_free,2),
        disk_total=round(disk_total,2),
        disk_percent=round(disk_percent,1),
        hourly_tokens=get_hourly_token_count(),
        hourly_history=get_hourly_history()
    )

@app.route('/chats', methods=['GET'])
def list_chats():
    return jsonify(chats=list(CHATS.values()))

@app.route('/chats', methods=['POST'])
def create_chat_api():
    return jsonify(chat=new_chat_record())

@app.route('/chats/<chat_id>', methods=['GET'])
def get_chat_api(chat_id):
    if chat_id not in CHATS:
        return jsonify(error='Chat not found'), 404
    return jsonify(chat=CHATS[chat_id])

@app.route('/chats/<chat_id>', methods=['DELETE'])
def delete_chat_api(chat_id):
    if chat_id not in CHATS:
        return jsonify(error='Chat not found'), 404
    del CHATS[chat_id]
    return jsonify(ok=True)

@app.route('/chat', methods=['POST'])
def chat():
    global MODEL_BUSY

    try:
        data = request.get_json(silent=True) or {}
        raw_messages = data.get('messages', [])
        if not raw_messages:
            return jsonify(error='No message received.'), 400

        formatted_messages = [{'role': m['role'], 'content': m['content']} for m in raw_messages[-12:]]
        chat_id = data.get('chat_id')
        current_chat = get_or_create_chat(chat_id)

        temperature = float(data.get('temperature', 0.70))
        top_p = float(data.get('top_p', 0.90))
        max_tokens = int(data.get('max_tokens', 200))
        repetition_penalty = float(data.get('repetition_penalty', 1.05))

        inputs = tokenizer.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
        elif hasattr(inputs, 'input_ids'):
            input_ids = inputs.input_ids
            attention_mask = getattr(inputs, 'attention_mask', None)
        else:
            input_ids = inputs
            attention_mask = None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        input_count = int(input_ids.shape[-1])
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        MODEL_BUSY = True
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )

        elapsed = max(time.perf_counter() - start_time, 0.001)
        MODEL_BUSY = False

        output_count = int(outputs.shape[-1] - input_count)
        tokens_per_second = output_count / elapsed

        response = tokenizer.decode(
            outputs[0][input_count:],
            skip_special_tokens=True
        ).strip()

        if not response:
            response = 'I could not generate a response. Please try again.'

        total = input_count + output_count
        now_str = datetime.now().strftime("%I:%M %p")

        record_token_usage(total)

        user_msg = raw_messages[-1]
        user_msg['timestamp'] = user_msg.get('timestamp', now_str)
        user_msg['tokens'] = user_msg.get('tokens', input_count)

        assistant_msg = {
            'role': 'assistant',
            'content': response,
            'timestamp': now_str,
            'tokens': output_count
        }

        current_chat['messages'] = raw_messages[:-1] + [user_msg, assistant_msg]
        current_chat['input_tokens'] += input_count
        current_chat['output_tokens'] += output_count
        current_chat['total_tokens'] += total
        current_chat['updated'] = datetime.now().isoformat(timespec='seconds')
        
        if len(current_chat['messages']) <= 3 and current_chat['name'].startswith('New Chat'):
            first = str(user_msg.get('content','')).strip().replace('\n',' ')
            current_chat['name'] = (first[:30] + '...') if len(first) > 30 else (first or current_chat['name'])

        return jsonify(
            response=response,
            timestamp=now_str,
            input_tokens=input_count,
            output_tokens=output_count,
            total_tokens=total,
            tokens_per_second=round(tokens_per_second, 2),
            hourly_tokens=get_hourly_token_count(),
            chat=current_chat
        )

    except torch.cuda.OutOfMemoryError:
        MODEL_BUSY = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return jsonify(error='GPU memory is full. Try a shorter conversation or lower Max tokens.'), 500
    except Exception as e:
        MODEL_BUSY = False
        traceback.print_exc()
        return jsonify(error=str(e)), 500

def open_browser():
    webbrowser.open(f'http://{HOST}:{PORT}')

if __name__ == '__main__':
    print("\n" + "="*55)
    print("                  NICK AI READY")
    print("="*55)
    print(f"\nOpen: http://{HOST}:{PORT}")
    print('Device:', DEVICE)
    print('GPU:', GPU_NAME)
    print('\nStarting web server...\n')
    threading.Timer(1.5, open_browser).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)