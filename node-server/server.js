import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
dotenv.config();

const app = express();
app.use(express.json());

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const MODEL = process.env.MODEL || "llama3";

app.post("/query", async (req, res) => {
  try {
    const { text, lang, user_id } = req.body;

    // First call your Python RAG retriever
    const retrRes = await fetch(`${process.env.PYTHON_ML_URL}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_k: 3, lang }),
    });
    const context = await retrRes.json();

    // Combine user query + retrieved docs
    const prompt = `
    You are an agriculture advisor.
    Question: ${text}
    Context: ${JSON.stringify(context)}
    Answer in ${lang}.
    `;

    // Call Ollama locally
    const ollamaRes = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: MODEL, prompt }),
    });

    // Ollama streams JSON lines → collect text
    let answer = "";
    const reader = ollamaRes.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = new TextDecoder().decode(value);
      for (const line of chunk.split("\n")) {
        if (!line.trim()) continue;
        const data = JSON.parse(line);
        if (data.response) answer += data.response;
      }
    }

    res.json({ answer, sources: context });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to query model" });
  }
});

app.listen(process.env.PORT || 3000, () =>
  console.log(`Node server running on ${process.env.PORT || 3000}`)
);
