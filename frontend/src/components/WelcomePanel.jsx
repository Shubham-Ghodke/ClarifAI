import React from 'react';
import { UploadIcon, BotIcon, SparklesIcon } from './icons';
import '../App.css';

const WelcomePanel = () => {
    return (
        <div className="welcome-panel-final" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '2rem', textAlign: 'center' }}>
            <div className="welcome-cta" style={{ maxWidth: '550px', width: '100%' }}>
                <div className="cta-content" style={{ display: 'flex', gap: '0.75rem', background: 'var(--bg-secondary)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', alignItems: 'center', textAlign: 'left' }}>
                    <div className="cta-icon" style={{ fontSize: '1.5rem' }}>💡</div>
                    <div className="cta-text" style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                        <strong>Quick Start:</strong> Upload one or more files in the <strong>Knowledge Base</strong> (the <strong>Files</strong> tab above or on the left), then start typing in the chat!
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WelcomePanel;
