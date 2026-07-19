import React, { useState, useEffect } from 'react';
import { UploadIcon, FileIcon, CheckIcon, TrashIcon } from './icons';
import '../App.css';

const FileUpload = () => {
    const [files, setFiles] = useState([]);
    const [status, setStatus] = useState("");
    const [uploading, setUploading] = useState(false);
    const [dragOver, setDragOver] = useState(false);
    const [documents, setDocuments] = useState([]);
    const [loadingDocs, setLoadingDocs] = useState(false);

    const fetchDocuments = async () => {
        setLoadingDocs(true);
        try {
            const response = await fetch("http://127.0.0.1:8000/documents");
            if (response.ok) {
                const data = await response.json();
                setDocuments(data);
            } else {
                console.error("Failed to fetch documents list");
            }
        } catch (error) {
            console.error("Error fetching documents:", error);
        } finally {
            setLoadingDocs(false);
        }
    };

    useEffect(() => {
        const initializeSession = async () => {
            const isSessionActive = sessionStorage.getItem("clarifai_session_active");
            if (!isSessionActive) {
                setLoadingDocs(true);
                try {
                    console.log("[SESSION] First time load, clearing previous documents...");
                    const response = await fetch("http://127.0.0.1:8000/documents", {
                        method: "DELETE"
                    });
                    if (response.ok) {
                        sessionStorage.setItem("clarifai_session_active", "true");
                    }
                } catch (error) {
                    console.error("Error clearing documents on first load:", error);
                }
            }
            fetchDocuments();
        };

        initializeSession();
    }, []);

    const handleFileChange = (e) => {
        if (e.target.files) {
            const selectedFiles = Array.from(e.target.files);
            // Add unique files to the upload queue
            setFiles(prev => {
                const fileMap = new Map(prev.map(f => [f.name, f]));
                selectedFiles.forEach(f => fileMap.set(f.name, f));
                return Array.from(fileMap.values());
            });
            setStatus("");
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragOver(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setDragOver(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);

        if (e.dataTransfer.files) {
            const allowedExtensions = ['.pdf', '.txt', '.doc', '.docx', '.xls', '.xlsx'];
            const droppedFiles = Array.from(e.dataTransfer.files).filter(f => {
                const name = f.name.toLowerCase();
                return allowedExtensions.some(ext => name.endsWith(ext));
            });
            if (droppedFiles.length > 0) {
                // Add unique dropped files to queue
                setFiles(prev => {
                    const fileMap = new Map(prev.map(f => [f.name, f]));
                    droppedFiles.forEach(f => fileMap.set(f.name, f));
                    return Array.from(fileMap.values());
                });
                setStatus("");
            } else {
                setStatus("Please upload only PDF, TXT, Word, or Excel files.");
            }
        }
    };

    const handleRemoveSelectedFile = (fileName) => {
        setFiles(prev => prev.filter(f => f.name !== fileName));
    };

    const handleUpload = async () => {
        if (files.length === 0) {
            setStatus("Please select at least one file first.");
            return;
        }

        setUploading(true);
        let successCount = 0;
        let totalChunks = 0;

        for (let i = 0; i < files.length; i++) {
            const fileToUpload = files[i];
            setStatus(`Uploading (${i + 1}/${files.length}): ${fileToUpload.name}...`);

            const formData = new FormData();
            formData.append("file", fileToUpload);

            try {
                const response = await fetch("http://127.0.0.1:8000/upload", {
                    method: "POST",
                    body: formData,
                });

                if (response.ok) {
                    const data = await response.json();
                    successCount++;
                    totalChunks += data.chunks;
                } else {
                    console.error(`Failed to upload ${fileToUpload.name}`);
                }
            } catch (error) {
                console.error(`Error uploading ${fileToUpload.name}:`, error);
            }
        }

        if (successCount === files.length) {
            setStatus(`Success! Uploaded ${successCount} files (${totalChunks} chunks processed).`);
            setFiles([]); // Clear upload list
            fetchDocuments(); // Refresh documents list
            setTimeout(() => setStatus(""), 4000);
        } else if (successCount > 0) {
            setStatus(`Partial success! Uploaded ${successCount} of ${files.length} files.`);
            // Keep failed files in queue
            setFiles(prev => prev.slice(successCount));
            fetchDocuments();
        } else {
            setStatus("Upload failed. Please try again.");
        }
        setUploading(false);
    };

    const handleDelete = async (filename) => {
        if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) {
            return;
        }

        try {
            setStatus(`Deleting ${filename}...`);
            const response = await fetch(`http://127.0.0.1:8000/documents/${encodeURIComponent(filename)}`, {
                method: "DELETE"
            });

            if (response.ok) {
                setStatus(`Success! Deleted ${filename}.`);
                fetchDocuments(); // Refresh list
                setTimeout(() => setStatus(""), 3000);
            } else {
                setStatus("Failed to delete document.");
            }
        } catch (error) {
            setStatus(`Error deleting: ${error.message}`);
        }
    };

    const handleDeleteAll = async () => {
        if (!window.confirm("Are you sure you want to delete ALL documents? This action cannot be undone.")) {
            return;
        }

        try {
            setStatus("Deleting all documents...");
            const response = await fetch("http://127.0.0.1:8000/documents", {
                method: "DELETE"
            });

            if (response.ok) {
                setStatus("Success! All documents deleted.");
                fetchDocuments(); // Refresh list
                setTimeout(() => setStatus(""), 3000);
            } else {
                setStatus("Failed to delete all documents.");
            }
        } catch (error) {
            setStatus(`Error deleting all: ${error.message}`);
        }
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    return (
        <div className="card file-upload-card">
            <h2 className="card-title">
                <UploadIcon size={24} />
                Knowledge Base
            </h2>
            <div className="file-upload-container">
                <div
                    className={`drag-drop-zone ${dragOver ? 'drag-over' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById('file-input').click()}
                >
                    <div className="upload-icon-wrapper">
                        <UploadIcon size={32} />
                    </div>
                    <p className="drag-drop-text">Drop your files here</p>
                    <p className="drag-drop-hint">or click to browse</p>
                    <p className="drag-drop-hint" style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
                        Supports PDF, TXT, Word, and Excel files (Multiple allowed)
                    </p>
                </div>

                <input
                    id="file-input"
                    type="file"
                    onChange={handleFileChange}
                    accept=".pdf,.txt,.doc,.docx,.xls,.xlsx"
                    className="file-input"
                    multiple
                />

                {/* Selected Files Queue */}
                {files.length > 0 && (
                    <div className="selected-files-list" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%', padding: '0.5rem 0' }}>
                        <p style={{ fontSize: '0.8125rem', fontWeight: '600', margin: '0 0 0.25rem 0', color: 'var(--text-primary)' }}>Files to upload:</p>
                        <div style={{ maxHeight: '150px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.5rem', paddingRight: '0.25rem' }}>
                            {files.map((selectedFile, index) => (
                                <div key={index} className="selected-file" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.5rem 0.75rem' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', overflow: 'hidden', flex: 1 }}>
                                        <FileIcon size={16} className="file-icon" style={{ flexShrink: 0 }} />
                                        <div className="file-name" style={{ fontSize: '0.8125rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '180px' }}>{selectedFile.name}</div>
                                    </div>
                                    <button 
                                        onClick={() => handleRemoveSelectedFile(selectedFile.name)}
                                        style={{ border: 'none', background: 'none', color: '#ef4444', cursor: 'pointer', padding: '0.125rem', display: 'flex', alignItems: 'center' }}
                                        title="Remove file"
                                    >
                                        <TrashIcon size={14} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <button
                    onClick={handleUpload}
                    disabled={uploading || files.length === 0}
                    className="btn btn-primary"
                >
                    {uploading ? (
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                            Uploading
                            <span style={{ display: 'inline-flex', gap: '0.15rem', alignItems: 'center', marginTop: '4px' }}>
                                <span className="dot" style={{ background: 'white', width: '4px', height: '4px', margin: 0 }}></span>
                                <span className="dot" style={{ background: 'white', width: '4px', height: '4px', margin: 0 }}></span>
                                <span className="dot" style={{ background: 'white', width: '4px', height: '4px', margin: 0 }}></span>
                            </span>
                        </span>
                    ) : (
                        <>
                            <CheckIcon size={18} />
                            Upload Documents
                        </>
                    )}
                </button>

                {status && (
                    <p className={`status-text ${status.includes('Success') ? 'success' : status.includes('Error') || status.includes('failed') ? 'error' : ''}`}>
                        {status}
                    </p>
                )}

                {/* Document Manager Section */}
                <div className="document-manager-section">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '0 0 0.75rem 0' }}>
                        <h3 className="card-title" style={{ fontSize: '1.05rem', margin: 0, fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <FileIcon size={18} />
                            Uploaded Files ({documents.length})
                        </h3>
                        {documents.length >= 2 && (
                            <button
                                onClick={handleDeleteAll}
                                className="doc-delete-btn"
                                title="Delete all documents"
                                style={{
                                    border: 'none',
                                    background: 'rgba(239, 68, 68, 0.1)',
                                    color: '#ef4444',
                                    cursor: 'pointer',
                                    padding: '0.35rem 0.6rem',
                                    borderRadius: '6px',
                                    fontSize: '0.75rem',
                                    fontWeight: '600',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.25rem',
                                    transition: 'background 0.2s'
                                }}
                            >
                                <TrashIcon size={12} />
                                Delete All
                            </button>
                        )}
                    </div>
                    {loadingDocs && documents.length === 0 ? (
                        <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>Checking for uploaded files...</p>
                    ) : documents.length === 0 ? (
                        <p style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)', fontStyle: 'italic', margin: 0 }}>
                            No files uploaded. Add some files above.
                        </p>
                    ) : (
                        <ul className="doc-list" style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            {documents.map((doc, idx) => (
                                <li key={idx} className="doc-item" style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '0.625rem' }}>
                                        <div style={{ display: 'flex', gap: '0.625rem', overflow: 'hidden', flex: 1 }}>
                                            <FileIcon size={18} style={{ color: 'var(--primary)', flexShrink: 0, marginTop: '0.125rem' }} />
                                            <div style={{ overflow: 'hidden', flex: 1 }}>
                                                <div className="doc-name" title={doc.name} style={{ fontWeight: '600', fontSize: '0.875rem', color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{doc.name}</div>
                                                <div className="doc-meta" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', marginTop: '0.35rem', fontSize: '0.75rem', alignItems: 'center' }}>
                                                    <span style={{ background: 'var(--bg-secondary)', padding: '0.125rem 0.375rem', borderRadius: '4px', border: '1px solid var(--border-color)', color: 'var(--text-secondary)', fontWeight: '500' }}>
                                                        {doc.type || 'TXT'}
                                                    </span>
                                                    <span style={{ color: 'var(--text-tertiary)' }}>•</span>
                                                    <span style={{ color: 'var(--text-secondary)' }}>{formatFileSize(doc.size)}</span>
                                                    <span style={{ color: 'var(--text-tertiary)' }}>•</span>
                                                    <span style={{ color: 'var(--text-secondary)' }}>{doc.chunks} chunks</span>
                                                </div>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => handleDelete(doc.name)}
                                            className="doc-delete-btn"
                                            title={`Delete ${doc.name}`}
                                            style={{ border: 'none', background: 'none', color: '#ef4444', cursor: 'pointer', padding: '0.25rem', display: 'flex', alignItems: 'center' }}
                                        >
                                            <TrashIcon size={16} />
                                        </button>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            </div>
        </div>
    );
};

export default FileUpload;
