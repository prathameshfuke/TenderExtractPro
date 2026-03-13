import { useState } from 'react';
import axios from 'axios';
import { Send, MessageSquareText } from 'lucide-react';

const api = axios.create({ baseURL: '/api' });

const starterQuestions = [
  'What are the main technical requirements?',
  'What is explicitly excluded from scope?',
  'What delivery or completion timeline is mentioned?',
];

export default function ChatPanel({ job }) {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const askQuestion = async (rawQuestion) => {
    const trimmed = rawQuestion.trim();
    if (!trimmed || !job?.job_id || loading) return;

    const userMessage = { role: 'user', text: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      const res = await api.post(`/jobs/${job.job_id}/ask`, { question: trimmed });
      const data = res.data || {};
      const botMessage = {
        role: 'assistant',
        text: data.answer || data.error || 'No answer returned.',
        confidence: data.confidence || 'LOW',
        citations: Array.isArray(data.citations) ? data.citations : [],
        error: data.error || null,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: 'Failed to query the document.',
          confidence: 'LOW',
          citations: [],
          error: err?.response?.data?.error || err.message,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div>
          <h3>Ask The Document</h3>
          <p>Answers are grounded only in retrieved chunks from this tender.</p>
        </div>
        <MessageSquareText size={18} />
      </div>

      {messages.length === 0 && (
        <div className="chat-starters">
          {starterQuestions.map((item) => (
            <button key={item} className="chat-chip" type="button" onClick={() => askQuestion(item)}>
              {item}
            </button>
          ))}
        </div>
      )}

      <div className="chat-thread">
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`chat-message ${message.role}`}>
            <div className="chat-bubble">
              <div className="chat-role">{message.role === 'user' ? 'You' : 'TenderExtractPro'}</div>
              <div>{message.text}</div>
              {message.role === 'assistant' && (
                <div className="chat-meta-row">
                  <span className={`badge ${String(message.confidence || '').toLowerCase() === 'high' ? 'high' : String(message.confidence || '').toLowerCase() === 'medium' ? 'medium' : 'low'}`}>
                    {message.confidence || 'LOW'}
                  </span>
                </div>
              )}
              {message.error && <div className="chat-error">{message.error}</div>}
              {Array.isArray(message.citations) && message.citations.length > 0 && (
                <div className="chat-citations">
                  {message.citations.map((citation, citationIndex) => (
                    <div key={`${citation.chunk_id}-${citationIndex}`} className="chat-citation-card">
                      <div className="chat-citation-meta">
                        <span>Page {citation.page || '-'}</span>
                        <span>{citation.chunk_id || 'NOT_FOUND'}</span>
                      </div>
                      <div className="chat-citation-quote">{citation.quote || 'NOT_FOUND'}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message assistant">
            <div className="chat-bubble">
              <div className="chat-role">TenderExtractPro</div>
              <div>Searching the document and drafting a grounded answer...</div>
            </div>
          </div>
        )}
      </div>

      <form
        className="chat-input-bar"
        onSubmit={(event) => {
          event.preventDefault();
          askQuestion(question);
        }}
      >
        <textarea
          className="chat-input"
          value={question}
          rows={3}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about scope, specs, timelines, exclusions, locations, compliance, pricing references..."
        />
        <button className="chat-send-btn" type="submit" disabled={loading || !question.trim()}>
          <Send size={16} />
          Ask
        </button>
      </form>
    </div>
  );
}