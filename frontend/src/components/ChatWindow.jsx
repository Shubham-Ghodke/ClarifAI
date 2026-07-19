import React, { useState, useRef, useEffect } from 'react';
import { SendIcon, BotIcon, UserIcon, TrashIcon, CopyIcon, CheckIcon } from './icons';
import WelcomePanel from './WelcomePanel';
import '../App.css';

const formatMessageLine = (line) => {
    if (!line) return "";
    
    // Check for legacy line-level 🎯
    const isLegacyHighlighted = line.includes('🎯') && !line.match(/🎯[^🎯]+🎯/);
    let cleanLine = line;
    if (isLegacyHighlighted) {
        cleanLine = line.replace('🎯', '').trim();
    }

    // Split by highlight ==text==, 🎯text🎯, or bold **text**
    const regex = /(==[^=]+==|🎯[^🎯]+🎯|\*\*[^*]+\*\*)/g;
    const parts = cleanLine.split(regex);
    
    const renderedParts = parts.map((part, idx) => {
        if (part.startsWith('==') && part.endsWith('==') && part.match(/^==[^=]+==$/)) {
            const content = part.slice(2, -2);
            return <mark key={idx} className="chat-highlight">{content}</mark>;
        } else if (part.startsWith('🎯') && part.endsWith('🎯') && part.match(/^🎯[^🎯]+🎯$/)) {
            const content = part.slice(1, -1);
            return <mark key={idx} className="chat-highlight">{content}</mark>;
        } else if (part.startsWith('**') && part.endsWith('**') && part.match(/^\*\*[^*]+\*\*$/)) {
            const content = part.slice(2, -2);
            return <strong key={idx}>{content}</strong>;
        }
        return part;
    });

    if (isLegacyHighlighted) {
        return <mark className="chat-highlight">{renderedParts}</mark>;
    }
    return renderedParts;
};

const getDisplayLines = (content) => {
    if (!content) return [];
    const lines = content.split('\n');
    const hasHighlights = content.includes('==') || content.includes('🎯');
    if (hasHighlights) {
        return lines.filter(line => line.includes('==') || line.includes('🎯'));
    }
    return lines;
};

const ChatWindow = () => {
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: 'Hello! I am ClarifAI, your intelligent AI document assistant. Ask me questions or request summaries based on your uploaded documents!',
            sources: [],
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [copiedIndex, setCopiedIndex] = useState(null);
    
    const messagesContainerRef = useRef(null);

    const scrollToBottom = (behavior = "smooth") => {
        setTimeout(() => {
            if (messagesContainerRef.current) {
                messagesContainerRef.current.scrollTo({
                    top: messagesContainerRef.current.scrollHeight,
                    behavior
                });
            }
        }, 100);
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, loading]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const userMessage = { role: 'user', content: input, timestamp };
        setMessages(prev => [...prev, userMessage]);
        setInput("");
        setLoading(true);

        try {
            const response = await fetch("http://127.0.0.1:8000/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ 
                    question: userMessage.content,
                    history: messages.map(msg => ({
                        role: msg.role,
                        content: msg.content
                    }))
                }),
            });

            const data = await response.json();
            const resTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            if (response.ok) {
                if (data.answers && Array.isArray(data.answers) && data.answers.length > 0) {
                    setMessages(prev => [
                        ...prev,
                        ...data.answers.map(ans => ({
                            role: 'assistant',
                            content: ans.answer,
                            logic: ans.logic,
                            sources: ans.sources || [],
                            timestamp: resTimestamp
                        }))
                    ]);
                } else {
                    setMessages(prev => [...prev, {
                        role: 'assistant',
                        content: data.answer,
                        logic: data.logic,
                        sources: data.sources || [],
                        timestamp: resTimestamp
                    }]);
                }
            } else {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: "Sorry, I encountered an error.",
                    timestamp: resTimestamp
                }]);
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "Error connecting to server.",
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleCopy = (content, index) => {
        navigator.clipboard.writeText(content);
        setCopiedIndex(index);
        setTimeout(() => setCopiedIndex(null), 2000);
    };

    const handleClearChat = () => {
        if (window.confirm('Are you sure you want to clear the chat history?')) {
            setMessages([{
                role: 'assistant',
                content: 'Chat cleared. How can I help you today?',
                sources: [],
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }]);
        }
    };

    const hasUserMessages = messages.some(msg => msg.role === 'user');

    return (
        <div className="chat-container card">
            <div className="chat-header">
                <div className="chat-header-content">
                    <BotIcon size={28} />
                    <div>
                        <h2>ClarifAI Agent</h2>
                    </div>
                </div>
                <button className="clear-chat-btn" onClick={handleClearChat}>
                    <TrashIcon size={16} />
                    Clear
                </button>
            </div>

            <div className="chat-messages" ref={messagesContainerRef}>
                {!hasUserMessages ? (
                    <WelcomePanel />
                ) : (
                    <>
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`message-wrapper ${msg.role}`}>
                                <div className={`message-avatar ${msg.role}`}>
                                    {msg.role === 'user' ? <UserIcon size={20} /> : <BotIcon size={20} />}
                                </div>
                                <div className="message-content">
                                    <div className={`message-bubble ${msg.role}`}>
                                        <div className="message-text">
                                            {getDisplayLines(msg.content).map((line, i, arr) => (
                                                <React.Fragment key={i}>
                                                    {formatMessageLine(line)}
                                                    {i < arr.length - 1 && <br />}
                                                </React.Fragment>
                                            ))}
                                        </div>

                                        {msg.sources && msg.sources.length > 0 && (
                                            <div className="sources">
                                                <strong>Sources:</strong>
                                                <div>
                                                    {msg.sources.map((source, i) => {
                                                        const cleanSource = source ? source.split('/').pop().split('\\').pop().replace(/^[0-9a-fA-F]{8}_/, '') : '';
                                                        return <span key={i} className="source-badge">{cleanSource}</span>;
                                                    })}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                    <div className="message-meta">
                                        <span className="message-timestamp">{msg.timestamp}</span>
                                        {msg.role === 'assistant' && (
                                            <div className="message-actions">
                                                <button
                                                    className="action-btn"
                                                    onClick={() => handleCopy(getDisplayLines(msg.content).join('\n'), idx)}
                                                >
                                                    {copiedIndex === idx ? (
                                                        <>
                                                            <CheckIcon size={12} />
                                                            Copied
                                                        </>
                                                    ) : (
                                                        <>
                                                            <CopyIcon size={12} />
                                                            Copy
                                                        </>
                                                    )}
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </>
                )}
                {loading && (
                    <div className="message-wrapper assistant">
                        <div className="message-avatar assistant">
                            <BotIcon size={20} />
                        </div>
                        <div className="message-content">
                            <div className="message-bubble assistant loading-bubble">
                                <div className="dot"></div>
                                <div className="dot"></div>
                                <div className="dot"></div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <div className="chat-input-area">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask a question..."
                    disabled={loading}
                />
                <button
                    onClick={handleSend}
                    disabled={loading || !input.trim()}
                    className="btn btn-primary send-btn"
                >
                    <SendIcon size={18} />
                    Send
                </button>
            </div>
        </div>
    );
};

export default ChatWindow;
