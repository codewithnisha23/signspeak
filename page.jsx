 "use client";
import {useCallback,useEffect,useRef,useState} from "react";
import {Hand,Camera,CameraOff,Volume2,Trash2,CheckCircle2,Lightbulb,Activity,History,ShieldCheck} from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const WS = API.replace(/^http/,"ws") + "/ws/recognize";
const CONNECTIONS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
export default function Home(){
 const videoRef=useRef(null),canvasRef=useRef(null),streamRef=useRef(null),wsRef=useRef(null),timerRef=useRef(null),busyRef=useRef(false);
 const [modelReady,setModelReady]=useState(false),[cameraOn,setCameraOn]=useState(false),[status,setStatus]=useState("Checking Python recognition model…"),[text,setText]=useState(""),[history,setHistory]=useState([]),[confidence,setConfidence]=useState(0),[hand,setHand]=useState(false),[error,setError]=useState("");
 const stopCamera=useCallback(()=>{if(timerRef.current)clearInterval(timerRef.current);timerRef.current=null;if(wsRef.current){wsRef.current.close();wsRef.current=null;}if(streamRef.current){streamRef.current.getTracks().forEach(t=>t.stop());streamRef.current=null;}if(videoRef.current)videoRef.current.srcObject=null;setCameraOn(false);setHand(false);setStatus("Camera stopped");},[]);
 useEffect(()=>{fetch(`${API}/api/status`).then(r=>r.json()).then(d=>{setModelReady(d.model_ready);setStatus(d.message);}).catch(()=>setStatus("Python backend is not connected. Start the backend server."));return()=>stopCamera();},[stopCamera]);
 const drawLandmarks=useCallback((pts)=>{const c=canvasRef.current,v=videoRef.current;if(!c||!v)return;const w=v.videoWidth||640,h=v.videoHeight||480;c.width=w;c.height=h;const x=c.getContext("2d");x.clearRect(0,0,w,h);if(!pts?.length)return;x.strokeStyle="#4df28c";x.lineWidth=3;x.fillStyle="#08c95b";CONNECTIONS.forEach(([a,b])=>{if(pts[a]&&pts[b]){x.beginPath();x.moveTo(pts[a].x*w,pts[a].y*h);x.lineTo(pts[b].x*w,pts[b].y*h);x.stroke();}});pts.forEach(p=>{x.beginPath();x.arc(p.x*w,p.y*h,4,0,Math.PI*2);x.fill();x.strokeStyle="#fff";x.lineWidth=1;x.stroke();});},[]);
 const startCamera=async()=>{
  setError("");
  if(!modelReady){setStatus("Model not trained yet. Follow the training instructions below.");return;}
  try{
   const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:720},facingMode:"user"},audio:false});
   streamRef.current=stream;videoRef.current.srcObject=stream;await videoRef.current.play();
   const ws=new WebSocket(WS);ws.binaryType="arraybuffer";wsRef.current=ws;
   ws.onopen=()=>{setCameraOn(true);setStatus("Camera running — show a trained gesture");timerRef.current=setInterval(async()=>{if(busyRef.current||ws.readyState!==WebSocket.OPEN||!videoRef.current?.videoWidth)return;busyRef.current=true;const c=document.createElement("canvas");c.width=640;c.height=480;c.getContext("2d").drawImage(videoRef.current,0,0,640,480);c.toBlob(async blob=>{try{if(blob&&ws.readyState===WebSocket.OPEN)ws.send(await blob.arrayBuffer());}finally{busyRef.current=false;}}, "image/jpeg",.72);},130);};
   ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.type==="ready"){setModelReady(d.model_ready);if(!d.model_ready)setStatus("Model not trained yet");return;}if(d.type==="result"){setHand(d.hand_detected);setConfidence(d.confidence||0);drawLandmarks(d.landmarks||[]);setStatus(d.message);if(d.accepted_text){setText(t=>t+(t?" ":"")+d.accepted_text);setHistory(h=>[{word:d.accepted_text,time:new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})},...h].slice(0,10));}}if(d.type==="error")setError(d.message);};
   ws.onerror=()=>setError("Could not connect to the Python recognition service.");
   ws.onclose=()=>{if(timerRef.current)clearInterval(timerRef.current);};
  }catch(e){setError(e.name==="NotAllowedError"?"Camera permission denied. Allow camera access and try again.":e.message||"Unable to start camera.");stopCamera();}
 };
 const speak=()=>{if(!text.trim())return;if("speechSynthesis"in window){window.speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang="en-IN";u.rate=.92;window.speechSynthesis.speak(u);}else setError("Speech synthesis is not supported in this browser.");};
 const clear=()=>{setText("");setStatus("Translated text cleared.");};
 return <main className="shell">
  <header className="hero"><div className="logo"><Hand size={34}/></div><div className="brand"><h1>SIGNSPEAK</h1><p>Sign Language Recognition and Translation System</p></div><div className="live"><i/> PYTHON AI ENGINE</div></header>
  <section className="mainGrid">
   <section className="panel cameraPanel"><div className="heading"><div><small>GESTURE INPUT</small><h2>Camera Preview</h2></div><span className={cameraOn?"state on":"state"}><i/>{cameraOn?"Camera running":"Camera stopped"}</span></div>
    <div className="stage"><video ref={videoRef} muted playsInline/><canvas ref={canvasRef}/>{!cameraOn&&<div className="placeholder"><Hand size={68}/><h3>Camera is stopped</h3><p>Start the camera and show a trained hand gesture.</p></div>}<div className="frame tl"/><div className="frame tr"/><div className="frame bl"/><div className="frame br"/></div>
    <div className="actions"><button className="primary" onClick={startCamera} disabled={cameraOn||!modelReady}><Camera size={19}/>Start Camera</button><button className="danger" onClick={stopCamera} disabled={!cameraOn}><CameraOff size={19}/>Stop Camera</button></div>
    <div className="status"><CheckCircle2 size={19}/><span>{error||status}</span></div>
   </section>
   <aside className="side">
    <section className="panel outputPanel"><div className="heading"><div><small>RECOGNITION OUTPUT</small><h2>Translated Text</h2></div><span className="aa">Aa</span></div><div className="translated"><small>DETECTED GESTURE TEXT</small><div>{text||"—"}</div><p>{text?`Last recognition confidence: ${(confidence*100).toFixed(1)}%`:"Recognized labels from your trained model will appear here."}</p></div><div className="actions"><button className="primary" onClick={speak} disabled={!text.trim()}><Volume2 size={18}/>Speak Text</button><button className="light" onClick={clear}><Trash2 size={18}/>Clear Text</button></div><div className="miniStatus"><Activity size={16}/>{modelReady?"Random Forest model loaded":"Model not trained yet"}</div></section>
    <section className="panel guide"><div className="heading"><div><small>HOW IT WORKS</small><h2>Recognition Guide</h2></div></div><div className="steps"><div><b>01</b><span>Camera captures your hand</span></div><div><b>02</b><span>MediaPipe extracts 21 landmarks</span></div><div><b>03</b><span>Random Forest predicts the gesture</span></div><div><b>04</b><span>Stable labels are added to text</span></div></div><div className="tip"><Lightbulb size={21}/><p><b>Tip:</b> Collect 50–100 samples per gesture with different hand positions and lighting, then train the model.</p></div></section>
   </aside>
  </section>
  <section className="lower"><section className="panel history"><div className="heading"><div><small>SESSION LOG</small><h2>Recent Translations</h2></div><button className="clearHistory" onClick={()=>setHistory([])}>Clear history</button></div>{history.length?<div className="historyItems">{history.map((h,i)=><span key={i}><b>{h.word}</b><small>{h.time}</small></span>)}</div>:<p className="empty">No gestures recognized yet.</p>}</section>
   <section className="panel note"><ShieldCheck size={25}/><div><b>Prototype information</b><p>This app uses your original Python recognition pipeline. It recognizes only gestures present in your trained dataset; it is not a complete sign-language translator.</p></div></section></section>
  <footer><b>✋ SIGNSPEAK</b><span>React frontend · FastAPI backend · MediaPipe · Random Forest</span></footer>
 </main>
}
