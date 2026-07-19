import React, { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import FileUpload from './components/FileUpload';
import ThemeToggle from './components/ThemeToggle';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-content">
          <h1>
            Welcome to ClarifAI
          </h1>
          <ThemeToggle />
        </div>
        <p className="header-subtitle">Your easy-to-use AI document assistant. Ask questions and get answers directly from your files!</p>
      </header>

      {/* Mobile/Tablet Tab Switcher */}
      <div className="mobile-tabs">
        <button 
          className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          💬 Chat
        </button>
        <button 
          className={`tab-btn ${activeTab === 'files' ? 'active' : ''}`}
          onClick={() => setActiveTab('files')}
        >
          📂 Files
        </button>
        <button 
          className={`tab-btn ${activeTab === 'about' ? 'active' : ''}`}
          onClick={() => setActiveTab('about')}
        >
          ✨ About
        </button>
      </div>

      <main className="main-content">
        <aside className={`aside-panel ${activeTab === 'files' ? 'mobile-visible' : 'mobile-hidden'}`}>
          <FileUpload />
        </aside>
        
        <section className="section-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className={`panel-wrapper ${activeTab === 'chat' ? 'mobile-visible' : 'mobile-hidden'}`}>
            <ChatWindow />
          </div>
          
          <div className={`panel-wrapper ${activeTab === 'about' ? 'mobile-visible' : 'mobile-hidden'}`}>
            <div className="clarifai-purpose-container">
              <div className="purpose-header">
                <h3 className="purpose-title">
                  What is ClarifAI?
                </h3>
                <p className="purpose-subtitle">
                  ClarifAI converts static files into an interactive knowledge engine, helping you understand, search, and extract insights from your documents effortlessly.
                </p>
              </div>

              <div className="purpose-grid">
                <div className="purpose-card">
                  <div className="purpose-card-icon">🔍</div>
                  <h4 className="purpose-card-title">Find Answers Instantly</h4>
                  <p className="purpose-card-description">
                    Ask questions in plain language and get direct, accurate answers without reading through long documents.
                  </p>
                </div>

                <div className="purpose-card">
                  <div className="purpose-card-icon">🌐</div>
                  <h4 className="purpose-card-title">Ask in Multiple Languages</h4>
                  <p className="purpose-card-description">
                    Ask questions in your preferred language—even if your documents are written in another language.
                  </p>
                </div>

                <div className="purpose-card">
                  <div className="purpose-card-icon">🔀</div>
                  <h4 className="purpose-card-title">Handle Multiple Questions</h4>
                  <p className="purpose-card-description">
                    Ask more than one question in a single message and receive clear answers for each one.
                  </p>
                </div>

                <div className="purpose-card">
                  <div className="purpose-card-icon">🛡️</div>
                  <h4 className="purpose-card-title">Verify Source Details</h4>
                  <p className="purpose-card-description">
                    Every answer shows you the exact document name and highlighted text lines so you can check the facts.
                  </p>
                </div>

                <div className="purpose-card">
                  <div className="purpose-card-icon">📂</div>
                  <h4 className="purpose-card-title">Work With All Your Files</h4>
                  <p className="purpose-card-description">
                    Upload multiple files at once and search through all of your documents together in one place.
                  </p>
                </div>

                <div className="purpose-card">
                  <div className="purpose-card-icon">✏️</div>
                  <h4 className="purpose-card-title">Smart Typo Tolerance</h4>
                  <p className="purpose-card-description">
                    Don't worry about minor spelling mistakes—ClarifAI understands what you mean even with typos.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
