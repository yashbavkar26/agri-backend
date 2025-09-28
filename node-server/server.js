require('dotenv').config(); 
const express = require('express'); 
const axios = require('axios'); 
const bodyParser = require('body-parser'); 
const multer = require('multer'); 
const fs = require('fs'); 
const path = require('path'); 
const crypto = require('crypto'); 
const { signPayload } = require('./helpers/signage'); 
const { PDFDocument, StandardFonts } = require('pdfkit'); // using pdfkit 
const { OpenAIApi, Configuration } = require("openai"); 
const sqlite3 = require('sqlite3').verbose(); 

const app = express(); 
app.use(bodyParser.json({limit: '5mb'})); 
app.use(bodyParser.urlencoded({ extended: true })); 
const upload = multer({ dest: 'uploads/' }); 
app.use(require('cors')()); 

const PYTHON_ML_URL = process.env.PYTHON_ML_URL || 'http://localhost:9000'; 
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || ''; 

if (!OPENAI_API_KEY) console.warn('OPENAI_API_KEY not set - LLM calls will fail.'); 

const openai = new OpenAIApi(new Configuration({ apiKey: OPENAI_API_KEY })); 

// Simple analytics DB (SQLite fallback) 
const DB_PATH = process.env.DB_PATH || path.join(__dirname, 'agri_analytics.db'); 
const db = new sqlite3.Database(DB_PATH); 
db.serialize(() => { 
  db.run(`CREATE TABLE IF NOT EXISTS queries ( 
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    ts TEXT, 
    user_id TEXT, 
    lang TEXT, 
    input_text TEXT, 
    answer_text TEXT 
  )`); 
}); 

// Utility: log query function 
function logQuery(user_id, lang, input_text, answer_text) { 
  const stmt = db.prepare("INSERT INTO queries (ts,user_id,lang,input_text,answer_text) VALUES (?,?,?,?,?)"); 
  stmt.run((new Date()).toISOString(), user_id || 'anon', lang || 'ml', input_text, answer_text); 
  stmt.finalize(); 
} 

// Helper: call python ML service to retrieve contexts 
async function retrieveContexts(text, lang, topK=4) { 
  // The Python ML service exposes /retrieve which returns an array of {id,title,excerpt,score} 
  const res = await axios.post(`${PYTHON_ML_URL}/retrieve`, { text, top_k: topK, lang }); 
  return res.data.results; 
} 

// Build RAG prompt for LLM 
function buildPrompt(query, lang, retrieved) { 
  let sources_text = retrieved.map(r => `ID: ${r.id}\nTitle: ${r.title}\nExcerpt: ${r.excerpt}\n`).join('\n---\n'); 
  const prompt = ` 
    You are AgriNova, a concise agricultural advisor for Kerala farmers. Language: ${lang} 
    User question: ${query} 
    You have access to these source excerpts: ${sources_text} 
    Deliver: 1) A short actionable answer (1-6 sentences) in the same language as the query. 2) One-line explanation why. 3) Cite source IDs used. 4) If uncertain, advise to consult local officer. Always add: "Follow label instructions and local govt. advisories." where pesticide/fungicide advice is given. 
  `; 
  return prompt; 
} 

// Endpoint: text query -> RAG -> LLM 
app.post('/query', async (req, res) => { 
  try { 
    const { text, lang = 'ml', user_id = 'anonymous', use_pinecone=false } = req.body; 
    if (!text) return res.status(400).json({ error: 'text required' }); 

    // 1) Retrieve contexts (Python ML service does FAISS or Pinecone) 
    const retrieved = await retrieveContexts(text, lang, 4); 

    // 2) Build prompt 
    const prompt = buildPrompt(text, lang, retrieved); 

    // 3) Call OpenAI Chat Completion (RAG) 
    // You may change model string to 'gpt-4' or 'gpt-4o-mini' depending availability 
    const chatResp = await openai.createChatCompletion({ 
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini', 
      messages: [ 
        { role: 'system', content: 'You are a concise agricultural advisor for Kerala farmers.' }, 
        { role: 'user', content: prompt } 
      ], 
      max_tokens: 500, 
      temperature: 0.0 
    }); 
    const answer = chatResp.data.choices[0].message.content; 

    // 4) Optional: request TTS generation from python ML service 
    let tts_filename = null; 
    try { 
      const ttsResp = await axios.post(`${PYTHON_ML_URL}/tts`, { text: answer, lang }, { responseType: 'json' }); 
      if (ttsResp.data && ttsResp.data.filename) tts_filename = ttsResp.data.filename; 
    } catch (e) { 
      console.warn('TTS generation failed (ok for demo):', e.message); 
    } 

    // 5) Log and respond 
    logQuery(user_id, lang, text, answer); 
    res.json({ input: text, answer, sources: retrieved, tts: tts_filename }); 
  } catch (err) { 
    console.error(err?.response?.data || err.message || err); 
    res.status(500).json({ error: 'server error', detail: err?.message || String(err) }); 
  } 
}); 

// Endpoint: upload audio -> python service transcribe -> run same pipeline 
app.post('/upload_audio', upload.single('audio'), async (req, res) => { 
  try { 
    const filePath = req.file.path; 
    const formData = new require('form-data')(); 
    formData.append('file', fs.createReadStream(filePath)); 
    formData.append('lang', req.body.lang || 'ml'); 

    const pyResp = await axios.post(`${PYTHON_ML_URL}/transcribe`, formData, { headers: formData.getHeaders() }); 
    fs.unlinkSync(filePath); 
    
    const transcription = pyResp.data.text; 

    // reuse /query logic by sending to our /query endpoint internally 
    const body = { 
      text: transcription, 
      lang: req.body.lang || 'ml', 
      user_id: req.body.user_id || 'anonymous' 
    }; 
    const internal = await axios.post(`http://localhost:${process.env.PORT || 3000}/query`, body); 

    res.json({ transcription, rag: internal.data }); 
  } catch (e) { 
    console.error(e); 
    res.status(500).json({ error: 'audio pipeline failed', detail: String(e.message || e) }); 
  } 
}); 

// Endpoint: certify -> sign JSON payload + optionally create PDF 
app.post('/certify', async (req, res) => { 
  try { 
    const { user_id='anonymous', query_text, lang='ml', answer_text, sources } = req.body; 
    if (!query_text || !answer_text) return res.status(400).json({ error: 'query_text and answer_text required' }); 

    const payload = { 
      user_id, 
      query_text, 
      lang, 
      answer_text, 
      sources, 
      issued_at: (new Date()).toISOString() 
    }; 

    // Sign payload with RSA private key (stored in certs/private.pem) 
    const signed = signPayload(payload); 

    // Create PDF 
    const pdfPath = path.join(__dirname, 'certs', `cert_${Date.now()}.pdf`); 
    const doc = new PDFDocument(); 
    doc.pipe(fs.createWriteStream(pdfPath)); 
    doc.fontSize(16).text('AI Farmer Advisor - Advisory Certificate', {underline:true}); 
    doc.moveDown(); 
    doc.fontSize(10).text(`Issued at: ${payload.issued_at}`); 
    doc.moveDown(); 
    doc.fontSize(12).text('Query:'); 
    doc.fontSize(10).text(query_text); 
    doc.moveDown(); 
    doc.fontSize(12).text('Answer:'); 
    doc.fontSize(10).text(answer_text); 
    doc.moveDown(); 
    doc.fontSize(12).text('Sources:'); 
    (sources || []).forEach(s => { 
      doc.fontSize(10).text(`${s.id || s.title || 'source'}: ${ (s.excerpt || '').slice(0,200) }`); 
    }); 
    doc.moveDown(); 
    doc.fontSize(8).text(`Signature (base64): ${signed.signature.slice(0,120)}...`); 
    doc.end(); 

    res.json({ signed, pdf: `/certs/${path.basename(pdfPath)}` }); 
  } catch (e) { 
    console.error(e); 
    res.status(500).json({ error: 'certify failed', detail: String(e.message || e) }); 
  } 
}); 

// Static serve certs folder 
app.use('/certs', express.static(path.join(__dirname, 'certs'))); 

// Simple analytics endpoints 
app.get('/analytics/top_queries', (req, res) => { 
  db.all("SELECT input_text, COUNT(*) as cnt FROM queries GROUP BY input_text ORDER BY cnt DESC LIMIT 20", (err, rows) => { 
    if (err) return res.status(500).json({ error: 'db error', detail: err.message }); 
    res.json({ top_queries: rows }); 
  }); 
}); 

app.get('/health', (req, res) => { 
  res.json({ status: 'ok', python_ml: PYTHON_ML_URL }); 
}); 

// ensure certs dir and RSA key exist 
if (!fs.existsSync(path.join(__dirname, 'certs'))) fs.mkdirSync(path.join(__dirname, 'certs')); 

if (!fs.existsSync(path.join(__dirname, 'certs/private.pem')) || !fs.existsSync(path.join(__dirname, 'certs/public.pem'))) { 
  // generate keys 
  const { generateKeyPairSync } = require('crypto'); 
  const { privateKey, publicKey } = generateKeyPairSync('rsa', { 
    modulusLength: 2048, 
  }); 
  fs.writeFileSync(path.join(__dirname, 'certs/private.pem'), privateKey.export({ type: 'pkcs1', format: 'pem' })); 
  fs.writeFileSync(path.join(__dirname, 'certs/public.pem'), publicKey.export({ type: 'spki', format: 'pem' })); 
} 

const PORT = process.env.PORT || 3000; 
app.listen(PORT, () => console.log(`Node server listening on ${PORT}`));